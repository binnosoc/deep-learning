input_file_db = "s11_param_db.s1p"
input_file_phase = "s11_param_phase.s1p"
output_file = "s11_param_cor.s1p"

with open(input_file_db, "r") as infile_db, open(input_file_phase, "r") as infile_phase, open(output_file, "w") as outfile:
    for line_db, line_phase in zip(infile_db, infile_phase):
        # Ignore les lignes de commentaires
        if line_db.startswith("!") or line_db.startswith("#"):
            outfile.write(line_db)
            
        else:
            values_db = line_db.split()
            values_phase = line_phase.split()
            freq_ghz = float(values_db[0]) / 1e9  # Conversion de Hz à GHz
            magnitude = float(values_db[1])
            phase = float(values_phase[1])
            outfile.write(f"{freq_ghz:.6f}\t{magnitude:.6e}\t{phase:.6}\n")
