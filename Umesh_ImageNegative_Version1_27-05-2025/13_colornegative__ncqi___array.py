import numpy as np
import time
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# --- Grayscale Pixel Negation ---
def negate_grayscale_pixel_quantum(pixel_value, intensity_bits=8, print_details=False, pixel_coords_for_print="Pixel"):
    if not (0 <= pixel_value < (1 << intensity_bits)):
        raise ValueError(f"Pixel value {pixel_value} is out of range for {intensity_bits} bits.")

    qr = QuantumRegister(intensity_bits, "intensity")
    cr = ClassicalRegister(intensity_bits, "measurement")
    qc = QuantumCircuit(qr, cr)

    # Encode value
    binary_representation = bin(pixel_value)[2:].zfill(intensity_bits)
    for j in range(intensity_bits):
        if binary_representation[intensity_bits - 1 - j] == '1':
            qc.x(qr[j])

    qc.barrier(label="Initial_State")
    qc.save_statevector(label="state_before_negation")

    # Bit-flip (negation)
    for j in range(intensity_bits):
        qc.x(qr[j])

    qc.barrier(label="After_Negation")
    qc.save_statevector(label="state_after_negation")
    qc.measure(qr, cr)

    if print_details:
        print(f"\n--- Quantum Details for {pixel_coords_for_print} (Input: {pixel_value}) ---")
        print(qc.draw(output="text"))

    backend = AerSimulator()
    transpiled_qc = transpile(qc, backend)
    job = backend.run(transpiled_qc)
    result = job.result()
    counts = result.get_counts(transpiled_qc)
    measured_bitstring = list(counts.keys())[0]
    negated_value = int(measured_bitstring, 2)

    return negated_value

# --- Color Pixel Negation ---
def negate_color_pixel_quantum(r_value, g_value, b_value, channel_bits=8, print_details=False, pixel_coords_for_print="Color Pixel"):
    negated_r = negate_grayscale_pixel_quantum(r_value, channel_bits, print_details=print_details,
                                               pixel_coords_for_print=f"{pixel_coords_for_print} - R")
    negated_g = negate_grayscale_pixel_quantum(g_value, channel_bits, print_details=print_details,
                                               pixel_coords_for_print=f"{pixel_coords_for_print} - G")
    negated_b = negate_grayscale_pixel_quantum(b_value, channel_bits, print_details=print_details,
                                               pixel_coords_for_print=f"{pixel_coords_for_print} - B")
    return negated_r, negated_g, negated_b

# --- Color Image Negation ---
def negate_color_image_quantum(image_matrix_rgb, channel_bits=8, max_circuit_prints=5):
    height = len(image_matrix_rgb)
    width = len(image_matrix_rgb[0])
    negated_image_matrix_rgb = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]

    print(f"\n--- Quantum Color Image Negation Started ({height}x{width}) ---")
    pixel_counter = 0

    for r_idx in range(height):
        for c_idx in range(width):
            r_val, g_val, b_val = image_matrix_rgb[r_idx][c_idx]
            print_this = pixel_counter < max_circuit_prints
            coords_str = f"Pixel ({r_idx},{c_idx})"
            negated_pixel = negate_color_pixel_quantum(r_val, g_val, b_val, channel_bits,
                                                       print_details=print_this, pixel_coords_for_print=coords_str)
            negated_image_matrix_rgb[r_idx][c_idx] = negated_pixel
            pixel_counter += 1

    return negated_image_matrix_rgb

# --- MSE Calculation ---
def calculate_mse(original_matrix, negated_matrix):
    original_np = np.array(original_matrix, dtype=np.int16)
    negated_np = np.array(negated_matrix, dtype=np.int16)
    mse = np.mean((original_np - negated_np) ** 2)
    return mse

# --- Main Execution ---
if __name__ == "__main__":
    channel_bits_color = 8
    color_image_example = [
        [(3, 0, 1), (0, 3, 2)],
        [(1, 1, 1), (2, 1, 0)]
    ]

    print(f"\nOriginal Color Image ({channel_bits_color}-bit per channel):")
    for row in color_image_example:
        print(row)

    # Classical Negation
    classical_negated = [[tuple(255 - val for val in pixel) for pixel in row] for row in color_image_example]

    # Quantum Negation
    start_time = time.time()
    quantum_negated = negate_color_image_quantum(color_image_example, channel_bits_color)
    end_time = time.time()

    print(f"\nQuantum Negated Color Image:")
    for row in quantum_negated:
        print(row)

    # MSE Comparison
    mse = calculate_mse(classical_negated, quantum_negated)
    print(f"\n📏 MSE between classical and quantum negation: {mse:.8f}")
    print(f"⏱️ Execution time: {end_time - start_time:.4f} seconds")
    print("\nNote: Circuits printed only for the first 5 pixels.")