import numpy as np
import time
import cv2
from qiskit import QuantumCircuit
from sklearn.metrics import mean_squared_error
from PIL import Image
import os

def ocqr_simulate_negate_pixel(r, g, b):
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    r_neg = (1 - r_norm) * 255
    g_neg = (1 - g_norm) * 255
    b_neg = (1 - b_norm) * 255
    return int(r_neg), int(g_neg), int(b_neg)

def classical_negate_image(image):
    norm = image / 255.0
    negated = 1 - norm
    return (negated * 255).astype(np.uint8)

def compute_mse(img1, img2):
    return mean_squared_error(img1.flatten(), img2.flatten())

def build_ocqr_circuit_for_5_pixels(pixels):
    qc = QuantumCircuit(15, name="OCQR_5_Pixels")
    for idx, (r, g, b) in enumerate(pixels):
        base_qubit = idx * 3
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
        theta_r = (1 - r_norm) * np.pi
        theta_g = (1 - g_norm) * np.pi
        theta_b = (1 - b_norm) * np.pi
        qc.ry(theta_r, base_qubit)
        qc.ry(theta_g, base_qubit + 1)
        qc.ry(theta_b, base_qubit + 2)
    return qc

def main():
    start_time = time.time()

    # === Update the image path here if needed ===
    img_path = 'lenna_original_512_color.tiff'  # Ensure this file exists

    # Load and convert TIFF image to RGB
    pil_img = Image.open(img_path).convert('RGB')
    pil_img = pil_img.resize((128, 128))
    image = np.array(pil_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f"Loaded and resized TIFF image to {image.shape}")

    quantum_simulated = np.zeros_like(image)

    # Get first 5 pixels RGB
    first_5_pixels = []
    count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if count < 5:
                b, g, r = image[i, j]
                first_5_pixels.append((r, g, b))
                count += 1
            else:
                break
        if count >= 5:
            break

    unified_circuit = build_ocqr_circuit_for_5_pixels(first_5_pixels)
    print("\nUnified Quantum Circuit (OCQR) for First 5 Pixels:\n")
    print(unified_circuit.draw(output='text'))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            b, g, r = image[i, j]
            qr, qg, qb = ocqr_simulate_negate_pixel(r, g, b)
            quantum_simulated[i, j] = [qb, qg, qr]  # BGR

    classical_negated = classical_negate_image(image)
    mse = compute_mse(classical_negated, quantum_simulated)
    print(f"\nMSE between Classical and OCQR Simulation: {mse:.6f}")
    print(f"Execution Time: {time.time() - start_time:.6f} seconds")

    # === Save the images ===
    os.makedirs("output_images", exist_ok=True)
    cv2.imwrite("output_images/original_image.png", image)
    cv2.imwrite("output_images/quantum_negated_image.png", quantum_simulated)
    print("\nImages saved to 'output_images/' folder.")

if __name__ == "__main__":
    main()
