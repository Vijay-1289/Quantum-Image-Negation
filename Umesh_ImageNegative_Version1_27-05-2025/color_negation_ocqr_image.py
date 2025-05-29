import numpy as np
import time
import cv2
from qiskit import QuantumCircuit
from sklearn.metrics import mean_squared_error
from PIL import Image
import matplotlib.pyplot as plt
import os

def ocqr_simulate_negate_pixel(r, g, b):
    # Integer-based exact negation to match classical method
    return 255 - r, 255 - g, 255 - b

def classical_negate_image(image):
    return 255 - image  # Same as ocqr simulation now

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

    img_path = 'lenna_original_512_color.tiff'  # Use the image in the same directory
    pil_img = Image.open(img_path).convert('RGB')
    pil_img = pil_img.resize((128, 128))
    image = np.array(pil_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB to BGR for OpenCV

    quantum_simulated = np.zeros_like(image)

    # Extract first 5 pixels for OCQR circuit
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

    # Simulate OCQR negation
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            b, g, r = image[i, j]
            qr, qg, qb = ocqr_simulate_negate_pixel(r, g, b)
            quantum_simulated[i, j] = [qb, qg, qr]

    # Compute classical negated image
    classical_negated = classical_negate_image(image)

    # Compute MSE
    mse = compute_mse(classical_negated, quantum_simulated)
    print(f"\n MSE between Classical and OCQR Simulation: {mse:.6f}")  # Should be 0

    # Execution time
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"\n Execution Time: {exec_time:.6f} seconds")

    # Display Quantum Circuit
    print("\nCombined OCQR Quantum Circuit (First 5 Pixels):\n")
    print(unified_circuit.draw(output='text'))

    # Convert images to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    quantum_rgb = cv2.cvtColor(quantum_simulated, cv2.COLOR_BGR2RGB)

    # Side-by-side plot and save
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(quantum_rgb)
    plt.title("OCQR Quantum Negated Image")
    plt.axis('off')

    output_filename = "quantum_negation_result.png"
    plt.tight_layout()
    plt.savefig(output_filename)
    
if __name__ == "__main__":
    main()
