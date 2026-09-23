import csv
import gc
import json
import random
import re
import time

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import networkx as nx

from skimage.morphology import skeletonize

import torch
import transformers
import tokenizers

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


# ==============================================================================
# 1. CONFIGURAÇÃO
# ==============================================================================

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

CONFIG = {
    # --------------------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------------------
    "train_questions_path": (
        "/kaggle/input/competitions/pcs-3838-pcs-5022-2026-parte-2/"
        "kaggle_dataset/train_dataset/text/questions_with_answers.jsonl"
    ),
    "train_images_dir": (
        "/kaggle/input/competitions/pcs-3838-pcs-5022-2026-parte-2/"
        "kaggle_dataset/train_dataset/images"
    ),
    "test_questions_path": (
        "/kaggle/input/competitions/pcs-3838-pcs-5022-2026-parte-2/"
        "kaggle_dataset/test_dataset/text/questions.jsonl"
    ),
    "test_images_dir": (
        "/kaggle/input/competitions/pcs-3838-pcs-5022-2026-parte-2/"
        "kaggle_dataset/test_dataset/images"
    ),

    # --------------------------------------------------------------------------
    # Modelo (usado apenas como fallback)
    # --------------------------------------------------------------------------
    "model_path": MODEL_PATH,
    "quantization": "8bit",
    "min_pixels": 256 * 28 * 28,
    "max_pixels": 1400 * 28 * 28,
    "count_max_new_tokens": 120,
    "simulation_max_new_tokens": 300,   # CoT precisa de mais tokens
    "count_samples": 1,
    "simulation_samples": 3,

    "lazy_load_vlm": True,

    # --------------------------------------------------------------------------
    # Execução
    # --------------------------------------------------------------------------
    "random_seed": 42,
    "val_fraction": 0.2,
    "validation_samples": 100,
    "main_mode": "validation",
    "test_samples": 5,

    # --------------------------------------------------------------------------
    # Cache e saída
    # --------------------------------------------------------------------------
    "count_cache_path": "/kaggle/working/count_cache_graph_v3.json",
    "simulation_cache_path": "/kaggle/working/simulation_cache_graph_v3.json",
    "graph_cache_path": "/kaggle/working/graph_cache_v3.json",
    "output_csv_path": "/kaggle/working/submission.csv",

    "save_cache_every": 10,
    "progress_every": 1,
}

random.seed(CONFIG["random_seed"])


# ==============================================================================
# 2. TIPOS DE PORTA
# ==============================================================================

VALID_GATE_TYPES = {"AND", "OR", "NOT", "NAND", "NOR", "XOR", "XNOR"}


def _eval_and(inputs):
    return all(inputs)


def _eval_or(inputs):
    return any(inputs)


def _eval_not(inputs):
    return not inputs[0]


def _eval_nand(inputs):
    return not _eval_and(inputs)


def _eval_nor(inputs):
    return not _eval_or(inputs)


def _eval_xor(inputs):
    return (sum(1 for value in inputs if value) % 2) == 1


def _eval_xnor(inputs):
    return not _eval_xor(inputs)


GATE_EVAL_FUNCTIONS = {
    "AND": _eval_and, "OR": _eval_or, "NOT": _eval_not,
    "NAND": _eval_nand, "NOR": _eval_nor, "XOR": _eval_xor, "XNOR": _eval_xnor,
}


# ==============================================================================
# 3. CONFIGURAÇÃO DE VISÃO COMPUTACIONAL
#
# Calibrado e validado manualmente contra uma imagem real de 13 portas
# (3 AND, 3 NOT, 2 NOR, 3 OR, 2 XOR) — classificação E topologia de
# conexões bateram 100%. Ainda assim, RODE debug_visualize_extraction()
# em imagens do SEU dataset antes de confiar cegamente: fontes de label,
# espessura de traço e estilo de "degrau" do fio podem variar.
# ==============================================================================

CONFIG_CV = {
    "use_otsu": True,
    "binary_threshold": 127,

    # Buracos candidatos a corpo de porta. Buracos de dígitos de texto
    # (labels x0, x1, Q, ...) tendem a ficar bem abaixo disso; corpos de
    # porta, bem acima. Se seu dataset tiver fontes de label muito grandes,
    # aumente esse valor até separar bem os dois grupos (use o script de
    # calibração no final deste arquivo, seção 7-debug).
    "gate_hole_min_area": 3000,
    "max_gate_area_fraction": 0.15,

    # NOT: triângulo — proporção altura/largura do bbox do buraco interno.
    "not_hw_ratio_min": 1.15,

    # Fração de preenchimento (área do buraco / área do bbox) separa
    # AND-family (D-shape, quase retangular) de OR-family (a curva de
    # entrada reduz a área).
    "and_fill_fraction_min": 0.85,

    # Bolha de inversão: buraco pequeno, quase circular, colado à borda
    # direita (saída) de um gate.
    "bubble_min_area": 350,
    "bubble_max_area": 950,
    "bubble_circularity_min": 0.80,
    "bubble_aspect_min": 0.65,
    "bubble_aspect_max": 1.55,
    "bubble_snap_distance": 22,

    # XOR/XNOR: curva extra de entrada, buscada varrendo linhas horizontais
    # perto do centro vertical do gate, à esquerda do seu bbox.
    "xor_strip_width": 45,
    "xor_strip_margin": 5,
    "xor_rows_to_sample": 7,
    "xor_rows_span": 12,
    "xor_min_positive_rows": 4,

    # Rastreamento de fio.
    "wire_dilate_kernel": 3,
    # Margem local de apagamento de cada porta (não confundir com o
    # parent_contour de tinta, que fica fundido a toda a rede de fios).
    "gate_erase_margin": 8,
    # Raio de busca para casar pontas de fio com portas. PRECISA ser >= à
    # maior margem de apagamento usada (XOR usa margin + 25 à esquerda),
    # senão a ponta sobrevivente fica fora de alcance.
    "terminal_snap_radius": 45,
}


# ==============================================================================
# 4. DETECÇÃO DE PORTAS
# ==============================================================================

class DetectedGate:
    __slots__ = (
        "gate_type", "hole_bbox", "parent_contour", "hole_contour",
        "input_points", "output_point", "confidence",
    )

    def __init__(self, gate_type, hole_bbox, parent_contour, hole_contour,
                 input_points, output_point, confidence):
        self.gate_type = gate_type
        self.hole_bbox = hole_bbox
        self.parent_contour = parent_contour
        self.hole_contour = hole_contour
        self.input_points = input_points
        self.output_point = output_point
        self.confidence = confidence

    @property
    def bbox(self):
        return self.hole_bbox

    @property
    def contour(self):
        return self.hole_contour


def _binarize(gray_image):
    if CONFIG_CV["use_otsu"]:
        _, binary = cv2.threshold(
            gray_image, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
    else:
        _, binary = cv2.threshold(
            gray_image, CONFIG_CV["binary_threshold"], 255,
            cv2.THRESH_BINARY_INV,
        )

    return binary


def _contour_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 0:
        return 0.0

    return float(4 * np.pi * area / (perimeter ** 2))


def _bbox_distance(bbox_a, bbox_b):
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    dx = max(ax - (bx + bw), bx - (ax + aw), 0)
    dy = max(ay - (by + bh), by - (ay + ah), 0)

    return (dx ** 2 + dy ** 2) ** 0.5


def _find_gate_holes(binary_image):
    """Retorna lista de (hole_contour, parent_index) para buracos grandes
    o suficiente para ser corpo de porta (RETR_CCOMP + hierarquia, não
    RETR_EXTERNAL — ver explicação no topo do arquivo)."""

    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None:
        return [], contours

    hierarchy = hierarchy[0]
    total_area = binary_image.shape[0] * binary_image.shape[1]
    max_area = total_area * CONFIG_CV["max_gate_area_fraction"]

    gate_holes = []

    for index, contour in enumerate(contours):
        parent = hierarchy[index][3]

        if parent == -1:
            continue  # não é um buraco, é um contorno de tinta externo

        area = cv2.contourArea(contour)

        if CONFIG_CV["gate_hole_min_area"] <= area <= max_area:
            gate_holes.append((contour, parent))

    return gate_holes, contours


def _find_small_holes(binary_image):
    """Candidatos a bolha de inversão (podem incluir buracos de texto —
    filtrados depois por proximidade a um gate)."""

    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    results = []

    for index, contour in enumerate(contours):
        if hierarchy[index][3] == -1:
            continue

        area = cv2.contourArea(contour)

        if CONFIG_CV["bubble_min_area"] <= area <= CONFIG_CV["bubble_max_area"]:
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h if h > 0 else 0

            if not (CONFIG_CV["bubble_aspect_min"] <= aspect <= CONFIG_CV["bubble_aspect_max"]):
                continue

            if _contour_circularity(contour) < CONFIG_CV["bubble_circularity_min"]:
                continue

            results.append((x, y, w, h))

    return results


def _has_bubble_near(gate_bbox, small_holes):
    x, y, w, h = gate_bbox

    for hole_bbox in small_holes:
        if _bbox_distance(gate_bbox, hole_bbox) <= CONFIG_CV["bubble_snap_distance"]:
            hole_cx = hole_bbox[0] + hole_bbox[2] / 2
            gate_right = x + w

            if hole_cx >= gate_right - 5:
                return True

    return False


def _has_xor_extra_curve(binary_image, hole_bbox):
    x, y, w, h = hole_bbox
    center_y = y + h // 2

    span = CONFIG_CV["xor_rows_span"]
    n_rows = CONFIG_CV["xor_rows_to_sample"]
    rows = np.linspace(center_y - span, center_y + span, n_rows).astype(int)

    strip_x0 = max(0, x - CONFIG_CV["xor_strip_width"])
    strip_x1 = x + CONFIG_CV["xor_strip_margin"]

    positive_rows = 0

    for row_y in rows:
        if row_y < 0 or row_y >= binary_image.shape[0]:
            continue

        row = binary_image[row_y, strip_x0:strip_x1]

        if np.any(row > 0):
            positive_rows += 1

    return positive_rows >= CONFIG_CV["xor_min_positive_rows"]


def detect_gates(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)

    gate_holes, contours = _find_gate_holes(binary)
    small_holes = _find_small_holes(binary)

    gates = []

    for hole_contour, parent_index in gate_holes:
        x, y, w, h = cv2.boundingRect(hole_contour)
        area = cv2.contourArea(hole_contour)

        hw_ratio = h / w if w > 0 else 0
        fill_fraction = area / (w * h) if (w * h) > 0 else 0

        has_bubble = _has_bubble_near((x, y, w, h), small_holes)

        if hw_ratio >= CONFIG_CV["not_hw_ratio_min"]:
            gate_type = "NOT"

        elif fill_fraction >= CONFIG_CV["and_fill_fraction_min"]:
            gate_type = "NAND" if has_bubble else "AND"

        else:
            is_xor_family = _has_xor_extra_curve(binary, (x, y, w, h))

            if is_xor_family:
                gate_type = "XNOR" if has_bubble else "XOR"
            else:
                gate_type = "NOR" if has_bubble else "OR"

        output_point = (x + w, y + h / 2.0)

        n_inputs = 1 if gate_type == "NOT" else 2
        input_points = [
            (x, y + h * (i + 1) / (n_inputs + 1))
            for i in range(n_inputs)
        ]

        confidence = fill_fraction if gate_type != "NOT" else hw_ratio

        gates.append(
            DetectedGate(
                gate_type=gate_type,
                hole_bbox=(x, y, w, h),
                parent_contour=contours[parent_index],
                hole_contour=hole_contour,
                input_points=input_points,
                output_point=output_point,
                confidence=confidence,
            )
        )

    gates.sort(key=lambda gate: gate.hole_bbox[0])

    return gates


# ==============================================================================
# 5. RASTREAMENTO DE FIOS (via pontas de esqueleto cortadas)
# ==============================================================================

def _mask_out_gate_bodies(binary_image, gates):
    """Apaga só a REGIÃO LOCAL de cada porta (bbox do buraco + margem
    pequena). Não usamos parent_contour: fio e porta são o mesmo
    componente de tinta conectado nesse dataset (sem espaço entre eles),
    então apagar o parent_contour apagaria a rede de fios inteira."""

    wires_only = binary_image.copy()
    height, width = binary_image.shape[:2]
    margin = CONFIG_CV["gate_erase_margin"]

    for gate in gates:
        x, y, w, h = gate.hole_bbox

        left_margin = margin + (25 if gate.gate_type in ("XOR", "XNOR") else 0)
        right_margin = margin + (15 if gate.gate_type in ("NAND", "NOR", "XNOR", "NOT") else 0)

        x0 = max(0, x - left_margin)
        x1 = min(width, x + w + right_margin)
        y0 = max(0, y - margin)
        y1 = min(height, y + h + margin)

        wires_only[y0:y1, x0:x1] = 0

    return wires_only


def _skeleton_components(binary_wires):
    skeleton = skeletonize(binary_wires > 0)
    ys, xs = np.nonzero(skeleton)

    pixel_graph = nx.Graph()
    coords_set = set(zip(xs.tolist(), ys.tolist()))
    pixel_graph.add_nodes_from(coords_set)

    neighbor_offsets = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
    ]

    for (px, py) in coords_set:
        for dx, dy in neighbor_offsets:
            neighbor = (px + dx, py + dy)

            if neighbor in coords_set:
                pixel_graph.add_edge((px, py), neighbor)

    components = {}

    for component_id, component in enumerate(nx.connected_components(pixel_graph)):
        for pixel in component:
            components[pixel] = component_id

    return coords_set, components


def _find_skeleton_endpoints(coords_set):
    """Pixels do esqueleto com exatamente 1 vizinho — pontas de fio
    soltas (cotos deixados pela apagação das portas)."""

    neighbor_offsets = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
    ]

    endpoints = []

    for (px, py) in coords_set:
        degree = 0

        for dx, dy in neighbor_offsets:
            if (px + dx, py + dy) in coords_set:
                degree += 1

                if degree > 1:
                    break

        if degree == 1:
            endpoints.append((px, py))

    return endpoints


def _assign_endpoints_to_gates(endpoints, gates, max_distance):
    """Associa cada ponta de fio solta ao gate mais próximo, marcando o
    lado (ENTRADA = esquerda do gate, SAÍDA = direita do gate). Retorna
    dict: gate_index -> {'input': [pixels], 'output': [pixels]}."""

    assignment = {i: {"input": [], "output": []} for i in range(len(gates))}

    for (ex, ey) in endpoints:
        best_gate = None
        best_side = None
        best_distance = max_distance

        for gate_index, gate in enumerate(gates):
            x, y, w, h = gate.hole_bbox

            if not (y - max_distance <= ey <= y + h + max_distance):
                continue

            if ex <= x:
                distance = ((x - ex) ** 2 + max(0, y - ey, ey - (y + h)) ** 2) ** 0.5
                side = "input"
            elif ex >= x + w:
                distance = ((ex - (x + w)) ** 2 + max(0, y - ey, ey - (y + h)) ** 2) ** 0.5
                side = "output"
            else:
                continue

            if distance < best_distance:
                best_distance = distance
                best_gate = gate_index
                best_side = side

        if best_gate is not None:
            assignment[best_gate][best_side].append((ex, ey))

    return assignment


def build_circuit_graph(image_bgr, gates):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)

    wires_only = _mask_out_gate_bodies(binary, gates)

    kernel = np.ones(
        (CONFIG_CV["wire_dilate_kernel"], CONFIG_CV["wire_dilate_kernel"]), np.uint8,
    )
    wires_only = cv2.dilate(wires_only, kernel, iterations=1)

    coords_set, components = _skeleton_components(wires_only)
    endpoints = _find_skeleton_endpoints(coords_set)
    endpoint_assignment = _assign_endpoints_to_gates(
        endpoints, gates, max_distance=CONFIG_CV["terminal_snap_radius"],
    )

    circuit_graph = nx.DiGraph()

    for gate_index, gate in enumerate(gates):
        circuit_graph.add_node(f"gate_{gate_index}", kind="gate", gate_type=gate.gate_type)

    # Duas pontas de fio no MESMO componente conexo do esqueleto são as
    # duas extremidades do mesmo fio físico -> conecta saída -> entrada.
    for source_index in range(len(gates)):
        output_pixels = endpoint_assignment[source_index]["output"]

        if not output_pixels:
            continue

        source_components = {components.get(px) for px in output_pixels}
        source_components.discard(None)

        for target_index in range(len(gates)):
            if target_index == source_index:
                continue

            for px in endpoint_assignment[target_index]["input"]:
                if components.get(px) in source_components:
                    circuit_graph.add_edge(f"gate_{source_index}", f"gate_{target_index}")
                    break

    # Entradas/pinos sem par (fio saindo pra fora do circuito) viram
    # entradas externas nomeadas x0, x1, ... por ordem vertical.
    unresolved_input_slots = []

    for gate_index, gate in enumerate(gates):
        n_predecessors = circuit_graph.in_degree(f"gate_{gate_index}")
        n_expected_inputs = len(gate.input_points)
        missing = n_expected_inputs - n_predecessors

        for _ in range(max(0, missing)):
            unresolved_input_slots.append((gate_index, gate.hole_bbox[1]))

    unresolved_input_slots.sort(key=lambda item: item[1])

    for order, (gate_index, _y) in enumerate(unresolved_input_slots):
        input_name = f"x{order}"
        circuit_graph.add_node(input_name, kind="input")
        circuit_graph.add_edge(input_name, f"gate_{gate_index}")

    sink_candidates = [
        node for node in circuit_graph.nodes
        if circuit_graph.nodes[node].get("kind") == "gate"
        and circuit_graph.out_degree(node) == 0
    ]

    if sink_candidates:
        sink_candidates.sort(
            key=lambda node: gates[int(node.split("_")[1])].hole_bbox[0],
            reverse=True,
        )

        circuit_graph.add_node("Q", kind="output")
        circuit_graph.add_edge(sink_candidates[0], "Q")

    return circuit_graph, []


# ==============================================================================
# 6. VALIDAÇÃO E AVALIAÇÃO DO GRAFO
# ==============================================================================

def validate_circuit_graph(circuit_graph, gates):
    if "Q" not in circuit_graph:
        return False, "nenhum nó de saída Q identificado"

    if not nx.is_directed_acyclic_graph(circuit_graph):
        return False, "grafo contém ciclo"

    for gate_index in range(len(gates)):
        node = f"gate_{gate_index}"

        if circuit_graph.in_degree(node) == 0 and circuit_graph.out_degree(node) == 0:
            return False, f"porta {node} totalmente desconectada"

    for gate_index, gate in enumerate(gates):
        node = f"gate_{gate_index}"

        if circuit_graph.in_degree(node) < 1:
            return False, f"porta {node} ({gate.gate_type}) sem entrada"

    return True, "ok"


def count_gates_from_graph(gates):
    counts = {gate_type: 0 for gate_type in VALID_GATE_TYPES}

    for gate in gates:
        counts[gate.gate_type] += 1

    return counts


def simulate_graph(circuit_graph, gates, assignments):
    values = dict(assignments)
    order = list(nx.topological_sort(circuit_graph))

    for node in order:
        if circuit_graph.nodes[node].get("kind") != "gate":
            continue

        gate_index = int(node.split("_")[1])
        gate_type = gates[gate_index].gate_type

        predecessor_values = [
            values[predecessor]
            for predecessor in circuit_graph.predecessors(node)
            if predecessor in values
        ]

        if not predecessor_values:
            raise RuntimeError(f"Entradas insuficientes para {node} ({gate_type}).")

        values[node] = GATE_EVAL_FUNCTIONS[gate_type](predecessor_values)

    q_predecessors = list(circuit_graph.predecessors("Q"))

    if not q_predecessors or q_predecessors[0] not in values:
        raise RuntimeError("Não foi possível resolver o valor de Q.")

    return "True" if values[q_predecessors[0]] else "False"


# ==============================================================================
# 7. DEPURAÇÃO VISUAL (rode isto ANTES da validação em massa)
# ==============================================================================

def debug_visualize_extraction(image_path, save_path=None):
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise FileNotFoundError(image_path)

    gates = detect_gates(image_bgr)
    circuit_graph, _ = build_circuit_graph(image_bgr, gates)
    is_valid, reason = validate_circuit_graph(circuit_graph, gates)

    overlay = image_bgr.copy()

    for gate_index, gate in enumerate(gates):
        x, y, w, h = gate.hole_bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay, f"{gate_index}:{gate.gate_type}", (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )

    print(f"Portas detectadas: {len(gates)}")
    print("Contagem por tipo:", dict(Counter(g.gate_type for g in gates)))
    print(f"Grafo válido: {is_valid} ({reason})")
    print("Arestas:", sorted(circuit_graph.edges))

    if save_path:
        cv2.imwrite(str(save_path), overlay)
        print(f"Overlay salvo em: {save_path}")

    return overlay, gates, circuit_graph, is_valid, reason


# ==============================================================================
# 8. MODELO QWEN (CARREGADO SOB DEMANDA, SÓ PARA FALLBACK)
# ==============================================================================

_LOCAL_MODEL = None
_LOCAL_PROCESSOR = None


def _ensure_vlm_loaded():
    global _LOCAL_MODEL, _LOCAL_PROCESSOR

    if _LOCAL_MODEL is not None:
        return

    print("\n[fallback] Carregando Qwen2.5-VL sob demanda...", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA não está disponível. Ative uma GPU Tesla T4.")

    model_path = CONFIG["model_path"]
    quantization = CONFIG["quantization"].lower()

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=CONFIG["min_pixels"],
        max_pixels=CONFIG["max_pixels"],
        use_fast=False,
    )

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }

    if quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    elif quantization != "fp16":
        raise ValueError('CONFIG["quantization"] deve ser "8bit", "4bit" ou "fp16".')

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    model.eval()

    _LOCAL_MODEL = model
    _LOCAL_PROCESSOR = processor
    print("[fallback] Qwen2.5-VL carregado.", flush=True)


def clear_gpu_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def call_vlm_with_image(image_path, prompt, max_new_tokens):
    _ensure_vlm_loaded()

    image_path = str(Path(image_path))

    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }]

    prompt_text = _LOCAL_PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = _LOCAL_PROCESSOR(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    model_device = next(_LOCAL_MODEL.parameters()).device
    inputs = inputs.to(model_device)

    generated_ids = None

    try:
        generated_ids = _LOCAL_MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=_LOCAL_PROCESSOR.tokenizer.pad_token_id,
            eos_token_id=_LOCAL_PROCESSOR.tokenizer.eos_token_id,
        )

        generated_only = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        result = _LOCAL_PROCESSOR.batch_decode(
            generated_only, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    finally:
        del inputs

        if generated_ids is not None:
            del generated_ids

        clear_gpu_memory()

    if not result:
        raise RuntimeError("O modelo retornou uma resposta vazia.")

    return result


COUNT_ALL_GATES_PROMPT = """
Inspect the complete logic circuit diagram and count every complete logic gate
by type.

A gate type may have count zero. Do not assume that any type is present.

GATE IDENTIFICATION:
- AND: flat left side and rounded right side, with no output bubble
- OR: curved input side and pointed output side, with no extra input curve
- NOT: triangle with exactly one small output bubble
- NAND: AND shape with exactly one small output bubble
- NOR: OR shape with exactly one small output bubble
- XOR: OR shape with one additional curved line on the input side
- XNOR: XOR shape with exactly one small output bubble

Return ONLY one valid JSON object with integer values.
Do not include Markdown or explanations.

{
  "AND": 0,
  "OR": 0,
  "NOT": 0,
  "NAND": 0,
  "NOR": 0,
  "XOR": 0,
  "XNOR": 0
}
"""

DIRECT_SIMULATION_PROMPT = """
You are an expert logic circuit simulator. Determine the Boolean value of output Q in the combinational logic circuit.

INPUT VALUES:
{assignments}

INSTRUCTIONS:
1. Identify the input values.
2. Trace the signals step-by-step through each logic gate (AND, OR, NOT, XOR, etc.).
3. Write down the intermediate output of each gate based on its inputs.
4. Continue until you evaluate the final gate connected to output Q.

After your step-by-step reasoning, you MUST write your final conclusion on the very last line exactly like this:
FINAL ANSWER: True
or
FINAL ANSWER: False
"""


def strip_json_fences(text):
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text):
    cleaned = strip_json_fences(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start < 0 or end <= start:
            raise

        return json.loads(cleaned[start:end + 1])


def normalize_gate_counts(raw_counts):
    result = {gate_type: 0 for gate_type in VALID_GATE_TYPES}

    if not isinstance(raw_counts, dict):
        return result

    for key, value in raw_counts.items():
        gate_type = str(key).strip().upper()

        if gate_type not in VALID_GATE_TYPES:
            continue

        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 0

        result[gate_type] = max(0, count)

    return result


def parse_boolean_answer(raw):
    cleaned = str(raw).strip()

    final_line_match = re.search(
        r"final answer\s*:\s*(true|false)", cleaned, flags=re.IGNORECASE,
    )

    if final_line_match:
        return final_line_match.group(1).capitalize()

    if cleaned.lower() == "true":
        return "True"

    if cleaned.lower() == "false":
        return "False"

    matches = re.findall(r"\b(true|false)\b", cleaned, flags=re.IGNORECASE)

    if not matches:
        return None

    return matches[-1].capitalize()


def format_assignments(assignments):
    ordered = sorted(assignments.items(), key=lambda item: int(item[0][1:]))

    return ", ".join(
        f"{variable}={'True' if value else 'False'}"
        for variable, value in ordered
    )


def fallback_count_all_gates(image_path):
    raw = call_vlm_with_image(
        image_path, COUNT_ALL_GATES_PROMPT, CONFIG["count_max_new_tokens"],
    )

    return normalize_gate_counts(extract_json_object(raw))


def fallback_simulate(image_path, assignments):
    prompt = DIRECT_SIMULATION_PROMPT.format(
        assignments=format_assignments(assignments)
    )

    predictions = []

    for _ in range(max(1, CONFIG["simulation_samples"])):
        raw = call_vlm_with_image(
            image_path, prompt, CONFIG["simulation_max_new_tokens"],
        )

        parsed = parse_boolean_answer(raw)

        if parsed is not None:
            predictions.append(parsed)

    if not predictions:
        raise RuntimeError("Fallback de simulação não produziu resposta válida.")

    frequencies = Counter(predictions)
    highest = max(frequencies.values())

    return next(value for value in predictions if frequencies[value] == highest)


# ==============================================================================
# 9. CARREGAMENTO DOS DADOS
# ==============================================================================

def load_jsonl(path):
    entries = []

    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                entries.append(json.loads(line))

    return entries


def image_path_for_index(images_dir, index):
    return str(Path(images_dir) / f"circuit_{index}.png")


# ==============================================================================
# 10. PARSER DE PERGUNTAS
# ==============================================================================

def parse_intent(question):
    normalized = " ".join(question.strip().split())
    lower = normalized.lower()

    gate_match = re.search(
        r"how many\s+(and|or|not|nand|nor|xor|xnor)\s+gates?", lower,
    )

    if gate_match:
        return {"operation": "count_gates", "gate_type": gate_match.group(1).upper()}

    if re.search(r"how many\s+(?:total\s+)?gates?", lower):
        return {"operation": "count_total_gates"}

    assignments = {}

    for variable, value in re.findall(
        r"\b(x\d+)\s*=\s*(true|false)\b", normalized, flags=re.IGNORECASE,
    ):
        assignments[variable.lower()] = (value.lower() == "true")

    if assignments:
        return {"operation": "simulate_output", "assignments": assignments}

    return {"operation": "other", "raw_question": question}


# ==============================================================================
# 11. FORMATAÇÃO DA RESPOSTA ESPERADA
# ==============================================================================

def format_answer(value):
    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, str):
        value = value.strip()

        if value.lower() in {"true", "false"}:
            return value.capitalize()

        number_match = re.search(r"\b\d+\b", value)

        if number_match:
            return number_match.group()

        return value

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


# ==============================================================================
# 12. CACHES
# ==============================================================================

GRAPH_CACHE = {}
COUNT_CACHE = {}
SIMULATION_CACHE = {}
_DIRTY = 0


def load_json_cache(path):
    path = Path(path)

    if not path.is_file():
        return {}

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_json_cache(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    temporary_path.replace(path)


def load_caches():
    global GRAPH_CACHE, COUNT_CACHE, SIMULATION_CACHE

    GRAPH_CACHE = load_json_cache(CONFIG["graph_cache_path"])
    COUNT_CACHE = load_json_cache(CONFIG["count_cache_path"])
    SIMULATION_CACHE = load_json_cache(CONFIG["simulation_cache_path"])

    print(f"Cache de grafos: {len(GRAPH_CACHE)}")
    print(f"Cache de contagens (fallback): {len(COUNT_CACHE)}")
    print(f"Cache de simulações (fallback): {len(SIMULATION_CACHE)}")


def save_caches():
    global _DIRTY

    save_json_cache(CONFIG["graph_cache_path"], GRAPH_CACHE)
    save_json_cache(CONFIG["count_cache_path"], COUNT_CACHE)
    save_json_cache(CONFIG["simulation_cache_path"], SIMULATION_CACHE)

    _DIRTY = 0


# ==============================================================================
# 13. EXTRAÇÃO DE GRAFO COM CACHE
# ==============================================================================

def _graph_cache_key(image_path):
    return f"{Path(image_path).resolve()}|graph-v3"


def get_circuit_extraction(image_path):
    global _DIRTY

    cache_key = _graph_cache_key(image_path)
    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    gates = detect_gates(image_bgr)
    circuit_graph, _ = build_circuit_graph(image_bgr, gates)
    is_valid, reason = validate_circuit_graph(circuit_graph, gates)

    if cache_key not in GRAPH_CACHE:
        GRAPH_CACHE[cache_key] = {
            "n_gates": len(gates),
            "is_valid": is_valid,
            "reason": reason,
            "counts": count_gates_from_graph(gates),
        }
        _DIRTY += 1

        if _DIRTY >= CONFIG["save_cache_every"]:
            save_caches()

    return gates, circuit_graph, is_valid


# ==============================================================================
# 14. RESPOSTA DE UMA QUESTÃO
# ==============================================================================

def answer_single_question(entry, images_dir):
    image_path = image_path_for_index(images_dir, entry["index"])
    intent = parse_intent(entry["question"])
    operation = intent["operation"]

    gates, circuit_graph, is_valid = get_circuit_extraction(image_path)

    if operation == "count_gates":
        if gates:
            counts = count_gates_from_graph(gates)
            return str(counts[intent["gate_type"]])

        counts = fallback_count_all_gates(image_path)
        return str(counts[intent["gate_type"]])

    if operation == "count_total_gates":
        if gates:
            counts = count_gates_from_graph(gates)
            return str(sum(counts.values()))

        counts = fallback_count_all_gates(image_path)
        return str(sum(counts.values()))

    if operation == "simulate_output":
        if is_valid:
            try:
                return simulate_graph(circuit_graph, gates, intent["assignments"])
            except Exception as exception:
                print(
                    f"  [AVISO] Simulação via grafo falhou "
                    f"({exception}); usando fallback.",
                    flush=True,
                )

        return fallback_simulate(image_path, intent["assignments"])

    raise RuntimeError(f"Operação não suportada: {operation}")


# ==============================================================================
# 15. PROGRESSO
# ==============================================================================

def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}min {seconds:02d}s"

    if minutes:
        return f"{minutes}min {seconds:02d}s"

    return f"{seconds}s"


def print_progress(position, total, entry, answer, sample_time, start_time):
    elapsed = time.perf_counter() - start_time
    average = elapsed / position
    remaining = average * (total - position)

    print(
        f"[{position}/{total}] index={entry['index']} answer={answer} "
        f"| amostra={sample_time:.1f}s | decorrido={format_duration(elapsed)} "
        f"| restante≈{format_duration(remaining)}",
        flush=True,
    )


# ==============================================================================
# 16. VALIDAÇÃO
# ==============================================================================

def run_validation(train_entries, images_dir, max_samples):
    entries = train_entries.copy()
    random.Random(CONFIG["random_seed"]).shuffle(entries)

    validation_size = max(1, int(len(entries) * CONFIG["val_fraction"]))
    entries = entries[:validation_size]

    if max_samples is not None:
        entries = entries[:max_samples]

    stats = defaultdict(lambda: [0, 0])
    graph_used_count = 0
    fallback_used_count = 0

    start_time = time.perf_counter()

    print(f"Iniciando validação com {len(entries)} amostras.", flush=True)

    try:
        for position, entry in enumerate(entries, start=1):
            sample_start = time.perf_counter()
            operation = parse_intent(entry["question"])["operation"]

            try:
                image_path = image_path_for_index(images_dir, entry["index"])
                _gates, _graph, is_valid = get_circuit_extraction(image_path)

                if is_valid:
                    graph_used_count += 1
                else:
                    fallback_used_count += 1

                predicted = answer_single_question(entry, images_dir)

            except Exception as exception:
                print(f"[AVISO] índice {entry['index']}: {exception}", flush=True)
                predicted = "0"

            expected = format_answer(entry["answer"])
            stats[operation][1] += 1

            if predicted == expected:
                stats[operation][0] += 1
            else:
                print(
                    f"\n[ERRO] index={entry['index']} | operação={operation}\n"
                    f"Pergunta: {entry['question']}\n"
                    f"Previsto: {predicted}\nEsperado: {expected}\n",
                    flush=True,
                )

            if position % CONFIG["progress_every"] == 0 or position == len(entries):
                print_progress(
                    position, len(entries), entry, predicted,
                    time.perf_counter() - sample_start, start_time,
                )

    finally:
        save_caches()

    print("\n=== Acurácia por operação ===")

    total_correct = 0
    total_count = 0

    for operation, (correct, count) in sorted(stats.items()):
        accuracy = correct / count if count else 0
        print(f"{operation:20s} {correct}/{count} = {accuracy:.2%}")
        total_correct += correct
        total_count += count

    overall = total_correct / total_count if total_count else 0
    print(f"\nAcurácia geral: {total_correct}/{total_count} = {overall:.2%}")

    print(
        f"\nGrafo válido (sem fallback): {graph_used_count} imagens | "
        f"Fallback usado: {fallback_used_count} imagens"
    )


# ==============================================================================
# 17. GERAÇÃO DA SUBMISSÃO
# ==============================================================================

def generate_submission(test_entries, images_dir, output_path, max_samples=None):
    entries = test_entries if max_samples is None else test_entries[:max_samples]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    print(f"Iniciando processamento de {len(entries)} amostras.", flush=True)

    try:
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["index", "answer"])
            output_file.flush()

            for position, entry in enumerate(entries, start=1):
                sample_start = time.perf_counter()

                try:
                    answer = answer_single_question(entry, images_dir)
                except Exception as exception:
                    print(f"[AVISO] índice {entry['index']}: {exception}", flush=True)
                    answer = "0"

                writer.writerow([entry["index"], answer])
                output_file.flush()

                if position % CONFIG["progress_every"] == 0 or position == len(entries):
                    print_progress(
                        position, len(entries), entry, answer,
                        time.perf_counter() - sample_start, start_time,
                    )

    finally:
        save_caches()

    elapsed = time.perf_counter() - start_time
    print(f"\nArquivo salvo em: {output_path}", flush=True)
    print(f"Tempo total: {format_duration(elapsed)}", flush=True)