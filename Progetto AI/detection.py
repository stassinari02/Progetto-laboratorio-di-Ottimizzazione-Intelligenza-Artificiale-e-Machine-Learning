import torch
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

# Funzione per calcolare l'Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1_topleft, y1_topleft, x1_bottomright, y1_bottomright = box1
    x2_topleft, y2_topleft, x2_bottomright, y2_bottomright = box2

    # Calcola le coordinate del rettangolo di intersezione
    x_inter_left = max(x1_topleft, x2_topleft)
    y_inter_top = max(y1_topleft, y2_topleft)
    x_inter_right = min(x1_bottomright, x2_bottomright)
    y_inter_bottom = min(y1_bottomright, y2_bottomright)

    if x_inter_right < x_inter_left or y_inter_bottom < y_inter_top:
        return 0.0

    # Calcola l'area di intersezione
    intersection_area = (x_inter_right - x_inter_left) * (y_inter_bottom - y_inter_top)

    # Calcola le aree dei bounding box
    area_box1 = (x1_bottomright - x1_topleft) * (y1_bottomright - y1_topleft)
    area_box2 = (x2_bottomright - x2_topleft) * (y2_bottomright - y2_topleft)

    # Calcola l'area dell'unione
    union_area = area_box1 + area_box2 - intersection_area

    # Calcola e restituisce IoU
    iou = intersection_area / union_area
    return iou

# Carica il modello pre-addestrato YOLOv5
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
print("Model loaded successfully")

# Definisci il percorso del dataset di immagini e il percorso per salvare i risultati
dataset_path = './archive/data/images'
output_path = './results'
os.makedirs(output_path, exist_ok=True)
print(f"Output directory created: {output_path}")

# Lista di immagini nel dataset
image_paths = list(Path(dataset_path).rglob('*.jpg')) + list(Path(dataset_path).rglob('*.png'))
print(f"Found {len(image_paths)} images in dataset")

# Carica il file CSV delle annotazioni
annotations_df = pd.read_csv('./archive/data/bounding_boxes.csv')
annotations_dict = {}

# Converti il DataFrame in un dizionario per accesso rapido
for idx, row in annotations_df.iterrows():
    image_id = row['image']
    bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
    if image_id not in annotations_dict:
        annotations_dict[image_id] = []
    annotations_dict[image_id].append({'bbox': bbox})

print("Annotations loaded successfully")

# Funzione di caricamento delle annotazioni di ground truth dal dizionario
def load_annotations(image_path):
    image_id = os.path.basename(image_path)
    return annotations_dict.get(image_id, [])

# Variabili per le statistiche
total_images = len(image_paths)
total_processed = 0
total_time = 0.0
total_tp = 0
total_fp = 0
total_fn = 0

# Funzione per visualizzare i risultati
def plot_results(img, results):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    for result in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = result
        label = f'{model.names[int(cls)]} {conf:.2f}'
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
        plt.gca().text(x1, y1, label, bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')
    plt.show()

# Processa ciascuna immagine
for image_path in image_paths:
    print(f'Processing image: {image_path}')
    img = cv2.imread(str(image_path))
    if img is None:
        print(f'Failed to load image: {image_path}')
        continue
    print("Image loaded successfully")

    # Misura il tempo di esecuzione per l'elaborazione dell'immagine
    start_time = time.time()
    results = model(img)
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_time += elapsed_time

    print(f"Detection completed for image: {image_path}")

    # Carica le annotazioni di ground truth per questa immagine
    annotations = load_annotations(image_path)
    print(f"Annotations for {image_path}: {annotations}")

    # Calcola true positive, false positive e false negative
    detected_boxes = []
    for result in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = result
        if model.names[int(cls)] == 'car':
            detected_boxes.append((x1, y1, x2, y2))

    print(f"Detected boxes: {detected_boxes}")

    matched_annotations = set()
    for detected_box in detected_boxes:
        is_tp = False
        for i, annotation in enumerate(annotations):
            if i in matched_annotations:
                continue
            iou = calculate_iou(detected_box, annotation['bbox'])
            print(f"IoU between detected box {detected_box} and annotation {annotation['bbox']} is {iou}")
            if iou > 0.5:
                total_tp += 1
                matched_annotations.add(i)
                is_tp = True
                break
        if not is_tp:
            total_fp += 1

    total_fn += len(annotations) - len(matched_annotations)
    total_processed += 1

    # Salva i risultati annotati
    annotated_img = results.render()[0]
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, annotated_img)
    print(f'Saved annotated image to: {output_image_path}')

    # Visualizza i risultati
    plot_results(img, results)
    print(f"Results plotted for image: {image_path}")

# Calcola le metriche finali
average_time_per_image = total_time / total_processed if total_processed > 0 else 0.0
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

# Stampa le statistiche finali
print(f"Total images : {total_images}")
print(f"Total images processed: {total_processed}")
print(f"Average time per image: {average_time_per_image:.2f} seconds")
print(f"Total True Positives (TP): {total_tp}")
print(f"Total False Positives (FP): {total_fp}")
print(f"Total False Negatives (FN): {total_fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")