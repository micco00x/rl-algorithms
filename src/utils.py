def save_performances(performances_file_name, performances, description=None):
    with open(performances_file_name, "w") as performances_file:
        if description:
            performances_file.write(description + "\n")
        for p in performances:
            performances_file.write("\t".join(p) + "\n")
