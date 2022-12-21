def update_table(line, name, file_name):
    with open(file_name, "a") as file:
        file.write(name + " & " + " & ".join(map(str, line)) + "\\")