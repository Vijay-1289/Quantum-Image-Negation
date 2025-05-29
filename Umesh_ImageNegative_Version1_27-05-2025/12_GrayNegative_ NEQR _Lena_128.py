from PIL import Image
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import time

# Convert integer to binary bit list
def int_to_bits(value, num_bits):
    return [int(bit) for bit in bin(value)[2:].zfill(num_bits)]

# Quantum negation of an image
def negate_image_quantum(image_path):
    try:
        img = Image.open(image_path).convert("L").resize((100, 100))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    width, height = img.size
    intensity_bits = 8
    matrix = [[img.getpixel((c, r)) for c in range(width)] for r in range(height)]
    classical_matrix = [[255 - matrix[r][c] for c in range(width)] for r in range(height)]
    negated_matrix = [[0 for _ in range(width)] for _ in range(height)]

    backend = AerSimulator()
    circuit_pairs = []

    print("\n--- Quantum Negation in Progress ---")
    start_time = time.time()
    pixel_count = 0

    for r in range(height):
        for c in range(width):
            qr = QuantumRegister(intensity_bits, 'q')
            cr = ClassicalRegister(intensity_bits, 'c')
            qc = QuantumCircuit(qr, cr)

            # Encode pixel
            bits = int_to_bits(matrix[r][c], intensity_bits)
            for j in range(intensity_bits):
                if bits[intensity_bits - 1 - j] == 1:
                    qc.x(qr[j])
            qc.barrier()

            # Quantum negation
            for j in range(intensity_bits):
                qc.x(qr[j])
            qc.barrier()
            qc.measure(qr, cr)

            # Simulate
            job = backend.run(qc, shots=1)
            result = job.result()
            bitstring = list(result.get_counts())[0]
            negated_value = int(bitstring, 2)
            negated_matrix[r][c] = negated_value

            # Store and print first 5 circuits
            if pixel_count < 5:
                input_qc = QuantumCircuit(qr, cr)
                for j in range(intensity_bits):
                    if bits[intensity_bits - 1 - j] == 1:
                        input_qc.x(qr[j])
                input_qc.barrier()
                circuit_pairs.append((input_qc, qc))
                pixel_count += 1

    end_time = time.time()
    execution_time = end_time - start_time

    # Compute MSE
    mse = np.mean((np.array(negated_matrix) - np.array(classical_matrix)) ** 2)

    # Build image from quantum-negated matrix
    neg_img = Image.new("L", (width, height))
    for r in range(height):
        for c in range(width):
            neg_img.putpixel((c, r), negated_matrix[r][c])

    return img, neg_img, circuit_pairs, mse, execution_time

# Main
if __name__ == "__main__":
    image_path = "Lenna_128_binary.tiff"  # Update this path

    result = negate_image_quantum(image_path)

    if result:
        orig_img, neg_img, circuits, mse, exec_time = result

        # Print quantum circuits for first 5 pixels
        for idx, (input_circuit, processed_circuit) in enumerate(circuits):
            print(f"\nInput Circuit for Pixel {idx + 1}:")
            print(input_circuit.draw(output="text"))
            print(f"\nProcessed Circuit for Pixel {idx + 1}:")
            print(processed_circuit.draw(output="text"))

        # Print metrics
        print(f"\n✅ MSE between Quantum and Classical Negation: {mse:.8f}")
        print(f"⏱️ Execution Time: {exec_time:.4f} seconds")

        # Display original and quantum negated image
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img.resize((128, 128)), cmap='gray')
        plt.title("Original (128x128)")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(neg_img.resize((128, 128)), cmap='gray')
        plt.title("Quantum Negated (128x128)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
