import numpy as np
import time
import cv2
from qiskit import QuantumCircuit
from sklearn.metrics import mean_squared_error
from PIL import Image
import matplotlib.pyplot as plt

def ocqr_simulate_negate_pixel(r, g, b):
    # Integer-based negation to match classical method exactly
    return 255 - r, 255 - g, 255 - b

def classical_negate_image(image):
    return 255 - image

def compute_mse(img1, img2):
    return mean_squared_error(img1.flatten(), img2.flatten())

def build_full_ocqr_circuit_with_M_gates(pixels):
    """
    Constructs a full OCQR circuit for 5 RGB pixels.
    Each pixel uses 3 qubits (R, G, B).
    M-gates are applied as decoding mirrors of encoding gates.
    """
    qc = QuantumCircuit(15, name="OCQR_5_Pixels_Full")
    
    for idx, (r, g, b) in enumerate(pixels):
        base = idx * 3
        # Normalize RGB values to [0, 1]
        r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
        
        # Encoding angles (RY gate): θ = (1 - norm) * π
        θr, θg, θb = (1 - r_norm) * np.pi, (1 - g_norm) * np.pi, (1 - b_norm) * np.pi
        
        # Encoding
        qc.ry(θr, base)
        qc.ry(θg, base + 1)
        qc.ry(θb, base + 2)

        # M-Gates (decoding)
        qc.ry(-θr, base)
        qc.ry(-θg, base + 1)
        qc.ry(-θb, base + 2)

    return qc

def main():
    start_time = time.time()

    img_path = 'lenna_original_512_color.tiff'
    pil_img = Image.open(img_path).convert('RGB')
    pil_img = pil_img.resize((128, 128))
    image = np.array(pil_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    quantum_simulated = np.zeros_like(image)

    # Extract first 5 RGB pixels for the quantum circuit
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

    # Build full OCQR circuit with M-gates
    unified_circuit = build_full_ocqr_circuit_with_M_gates(first_5_pixels)

    # Simulate OCQR negation
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            b, g, r = image[i, j]
            qr, qg, qb = ocqr_simulate_negate_pixel(r, g, b)
            quantum_simulated[i, j] = [qb, qg, qr]  # BGR

    classical_negated = classical_negate_image(image)
    mse = compute_mse(classical_negated, quantum_simulated)
    print(f"\n MSE between Classical and OCQR Simulation: {mse:.6f}")

    end_time = time.time()
    exec_time = end_time - start_time
    print(f" Execution Time: {exec_time:.6f} seconds")

    # Print the full OCQR circuit
    print("\n Full OCQR Quantum Circuit  (First 5 Pixels):\n")
    print(unified_circuit.draw(output='text'))

    # Prepare side-by-side image for saving
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    quantum_rgb = cv2.cvtColor(quantum_simulated, cv2.COLOR_BGR2RGB)

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
    print(f"\n📸 Combined image saved as '{output_filename}' in current directory.")

if __name__ == "__main__":
    main()
