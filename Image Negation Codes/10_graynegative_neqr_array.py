import numpy as np
import time
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

# --- Grayscale Pixel Negation using Quantum Circuit ---
def negate_grayscale_pixel_quantum(pixel_value, intensity_bits=8, print_details=False, pixel_coords_for_print="Pixel"):
    if not (0 <= pixel_value < (1 << intensity_bits)):
        raise ValueError(f"Pixel value {pixel_value} is out of range for {intensity_bits} bits.")

    qr = QuantumRegister(intensity_bits, "intensity")
    cr = ClassicalRegister(intensity_bits, "measurement")
    qc = QuantumCircuit(qr, cr)

    binary_rep = bin(pixel_value)[2:].zfill(intensity_bits)
    for j in range(intensity_bits):
        if binary_rep[intensity_bits - 1 - j] == '1':
            qc.x(qr[j])

    qc.barrier(label="Initial_State")
    qc.save_statevector(label="state_before_negation")

    for j in range(intensity_bits):
        qc.x(qr[j])

    qc.barrier(label="After_Negation")
    qc.save_statevector(label="state_after_negation")
    qc.measure(qr, cr)

    if print_details:
        print(f"\n--- {pixel_coords_for_print} (Input: {pixel_value}) ---")
        print(qc.draw(output="text"))

    backend = AerSimulator()
    transpiled_qc = transpile(qc, backend)
    job = backend.run(transpiled_qc)
    result = job.result()
    counts = result.get_counts(transpiled_qc)
    measured_bitstring = list(counts.keys())[0]
    negated_value = int(measured_bitstring, 2)

    return negated_value

# --- Classical Negation ---
def classical_negation(image_matrix, intensity_bits):
    max_val = (1 << intensity_bits) - 1
    return [[max_val - pixel for pixel in row] for row in image_matrix]

# --- Quantum Negation for Entire Image ---
def negate_grayscale_image_quantum(image_matrix, intensity_bits=8):
    height = len(image_matrix)
    width = len(image_matrix[0])
    negated_image_matrix = [[0 for _ in range(width)] for _ in range(height)]

    print(f"\n--- Quantum Negation of Grayscale Image ({height}x{width}, {intensity_bits}-bit) ---")

    print_count = 0
    for r in range(height):
        for c in range(width):
            original_pixel = image_matrix[r][c]
            print_flag = print_count < 5  # Print only for first 5 pixels
            negated_pixel = negate_grayscale_pixel_quantum(
                original_pixel,
                intensity_bits,
                print_details=print_flag,
                pixel_coords_for_print=f"Pixel ({r},{c})"
            )
            negated_image_matrix[r][c] = negated_pixel
            print_count += 1

    return negated_image_matrix

# --- Mean Squared Error ---
def mean_squared_error(img1, img2):
    return np.mean((np.array(img1, dtype=int) - np.array(img2, dtype=int)) ** 2)

# --- Main Execution ---
if __name__ == "__main__":
    grayscale_image = [
        [1, 6],
        [3, 5]
    ]
    intensity_bits = 3

    print(f"\nOriginal Grayscale Image ({intensity_bits}-bit):")
    for row in grayscale_image:
        print(row)

    classical_negated = classical_negation(grayscale_image, intensity_bits)

    # Measure execution time
    start_time = time.time()
    quantum_negated = negate_grayscale_image_quantum(grayscale_image, intensity_bits)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nQuantum Negated Image:")
    for row in quantum_negated:
        print(row)

    mse = mean_squared_error(classical_negated, quantum_negated)
    print(f"\n Mean Squared Error (MSE): {mse:.2f}")
    print(f"Quantum Negation Execution Time: {execution_time:.4f} seconds")