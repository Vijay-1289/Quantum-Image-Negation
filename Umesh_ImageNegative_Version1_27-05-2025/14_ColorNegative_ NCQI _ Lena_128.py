import time
from PIL import Image
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# Convert integer to binary (MSB first)
def int_to_bits(value, num_bits):
    return [int(bit) for bit in format(value, f"0{num_bits}b")]

# Properly reverse bits from Qiskit output (little-endian to big-endian)
def reverse_bit_order(bitstring, total_bits):
    return bitstring[::-1]  # global flip since all are measured

# Quantum negation with fixed bit ordering
def negate_image_quantum(image_path):
    img = Image.open(image_path).convert("RGB").resize((100, 100))
    width, height = img.size
    bits = 8

    matrix = [[img.getpixel((c, r)) for c in range(width)] for r in range(height)]
    classical_matrix = [[tuple(255 - v for v in pixel) for pixel in row] for row in matrix]
    negated_matrix = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]

    backend = AerSimulator()
    circuits_input = []
    circuits_processed = []

    pixel_count = 0
    start_time = time.time()

    for r in range(height):
        for c in range(width):
            r_val, g_val, b_val = matrix[r][c]
            qr = QuantumRegister(24, 'q')
            cr = ClassicalRegister(24, 'c')
            qc = QuantumCircuit(qr, cr)

            # Encode and negate each channel
            for idx, val in enumerate((r_val, g_val, b_val)):
                binary = int_to_bits(val, bits)
                offset = idx * bits
                for j in range(bits):
                    if binary[j] == 1:  # MSB first
                        qc.x(qr[offset + j])
                qc.barrier()
                for j in range(bits):
                    qc.x(qr[offset + j])  # Apply X to flip
            qc.barrier()
            qc.measure(qr, cr)

            # Save first 5 pixel circuits
            if pixel_count < 5:
                circuits_input.append(qc.decompose().copy(name="Input"))
                circuits_processed.append(qc)

            # Run and interpret
            job = backend.run(qc, shots=1)
            raw_bitstring = list(job.result().get_counts().keys())[0]
            corrected = reverse_bit_order(raw_bitstring, 24)

            r_bin = corrected[0:8]
            g_bin = corrected[8:16]
            b_bin = corrected[16:24]

            r_neg = int(r_bin, 2)
            g_neg = int(g_bin, 2)
            b_neg = int(b_bin, 2)

            negated_matrix[r][c] = (r_neg, g_neg, b_neg)
            pixel_count += 1

    end_time = time.time()
    mse = np.mean((np.array(classical_matrix, dtype=np.int16) - np.array(negated_matrix, dtype=np.int16)) ** 2)

    neg_img = Image.new("RGB", (width, height))
    for r in range(height):
        for c in range(width):
            neg_img.putpixel((c, r), negated_matrix[r][c])

    return matrix, negated_matrix, circuits_input, circuits_processed, img, neg_img, mse, end_time - start_time

# MAIN
if __name__ == "__main__":
    image_path = "lenna_original_512_color.tiff"
    print("🧪 Running quantum image negation...")

    result = negate_image_quantum(image_path)
    if result[0] is None:
        print("❌ Image processing failed.")
    else:
        matrix, negated_matrix, circuits_input, circuits_processed, orig_img, neg_img, mse, exec_time = result

        for i in range(len(circuits_input)):
            print(f"\n🔹 Circuit {i+1} (Input):\n{circuits_input[i].draw(output='text')}")
            print(f"\n🔹 Circuit {i+1} (Processed):\n{circuits_processed[i].draw(output='text')}")

        # Show results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img.resize((128, 128)))
        plt.title("Original (128x128)")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(neg_img.resize((128, 128)))
        plt.title("Quantum Negated (128x128)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        print(f"\n📏 MSE: {mse:.6f}")
        print(f"⏱️ Execution Time: {exec_time:.2f} seconds")
