"""Creates two sample physics exam PDFs in data/ so we can test the pipeline."""
import os
import fitz  # PyMuPDF

DATA = os.path.join(os.path.dirname(__file__), "data")

papers = {
    "physics_2019.pdf": (
        "PHYSICS - Annual Examination 2019\nTime: 3 hours       Max Marks: 30\n"
        "Answer all questions.\n",
        [
            "Q1. (5 marks) State the first law of thermodynamics and derive the relation between Cp and Cv for an ideal gas.",
            "Q2. (5 marks) Explain total internal reflection with a neat diagram and give two practical applications.",
            "Q3. (10 marks) Using Ampere's circuital law, derive an expression for the magnetic field due to a long straight current-carrying conductor.",
            "Q4. (5 marks) A body is projected at 30 degrees to the horizontal with speed 20 m/s. Find its range and maximum height. (Mechanics)",
            "Q5. (5 marks) Explain the photoelectric effect and write Einstein's photoelectric equation.",
        ],
    ),
    "physics_2020.pdf": (
        "PHYSICS - Annual Examination 2020\nTime: 3 hours       Max Marks: 30\n"
        "Answer all questions.\n",
        [
            "Q1. (5 marks) Define entropy and state the second law of thermodynamics.",
            "Q2. (5 marks) Derive the lens maker's formula for a thin convex lens.",
            "Q3. (10 marks) State and explain Faraday's laws of electromagnetic induction with an experiment.",
            "Q4. (5 marks) State the principle of conservation of angular momentum and explain with an example. (Mechanics)",
            "Q5. (5 marks) Describe Bohr's model of the hydrogen atom and derive the expression for the radius of the nth orbit.",
        ],
    ),
}

for name, (header, questions) in papers.items():
    doc = fitz.open()
    page = doc.new_page()
    body = header + "\n\n" + "\n\n".join(questions)
    page.insert_textbox(fitz.Rect(50, 50, 545, 800), body, fontsize=12, fontname="helv")
    doc.save(os.path.join(DATA, name))
    doc.close()
    print("Created", name)

print("Done.")
