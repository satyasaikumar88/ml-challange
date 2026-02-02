with open("magnitude_report.txt", "r") as f:
    for line in f:
        if "Average Change" in line or "restoring variance" in line or "Variance Check" in line:
            print(line.strip())
