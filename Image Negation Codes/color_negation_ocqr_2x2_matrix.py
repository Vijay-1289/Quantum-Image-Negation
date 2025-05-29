import numpy as np
import time
from qiskit import QuantumCircuit
from sklearn.metrics import mean_squared_error

# ----- OCQR Encode using RY gates -----
def ocqr_gate_encoding(r, g, b, pixel_index):
    # Normalize to [0,1]
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    # Convert to theta = (1 - x) * pi (negated value)
    r_theta = (1 - r_norm) * np.pi
    g_theta = (1 - g_norm) * np.pi
    b_theta = (1 - b_norm) * np.pi

    # Create a 3-qubit quantum circuit
    qc = QuantumCircuit(3, name=f"Pixel-{pixel_index}")
    qc.ry(r_theta, 0)
    qc.ry(g_theta, 1)
    qc.ry(b_theta, 2)

    print(f"\n🧠 OCQR Quantum Circuit using RY gates for Pixel {pixel_index} RGB({r},{g},{b}):")
    print(qc.draw(output='text'))

# ----- Classical OCQR-style Negation -----
def classical_ocqr_negate_pixel(r, g, b):
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    return int((1 - r_norm) * 255), int((1 - g_norm) * 255), int((1 - b_norm) * 255)

# ----- Classical Full Image Negation -----
def classical_negate_image(image):
    normalized = image / 255.0
    negated = 1.0 - normalized
    return (negated * 255).astype(np.uint8)

# ----- MSE -----
def compute_mse(img1, img2):
    return mean_squared_error(img1.flatten(), img2.flatten())

# ----- Main -----
def main():
    start = time.time()

    # Small test image (2x3 pixels)
    original_image = np.array([
        [[100, 150, 200], [50, 100, 150], [200, 50, 100]],
        [[25, 75, 125], [0, 255, 128], [255, 0, 64]]
    ], dtype=np.uint8)

    print("Original Image (2x3):\n", original_image)

    classical_negated = classical_negate_image(original_image)
    quantum_simulated = np.zeros_like(original_image)

    pixel_count = 0
    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            r, g, b = original_image[i, j]
            if pixel_count < 5:
                ocqr_gate_encoding(r, g, b, pixel_count + 1)
            qr, qg, qb = classical_ocqr_negate_pixel(r, g, b)
            quantum_simulated[i, j] = [qr, qg, qb]
            pixel_count += 1

    print("\nSimulated Quantum (OCQR) Negated Image:\n", quantum_simulated)

    mse = compute_mse(classical_negated, quantum_simulated)
    print("\nMSE between Classical and OCQR Simulated Image: {:.4f}".format(mse))

    end = time.time()
    print("\nExecution Time: {:.6f} seconds".format(end - start))

# Run
if __name__ == "__main__":
    main()
