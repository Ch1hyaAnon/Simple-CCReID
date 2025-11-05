from pathlib import Path

ROOT = Path(__file__).resolve().parent
all_path = ROOT / "all.txt"
train_path = ROOT / "train.txt"
gallery_path = ROOT / "gallery.txt"
query_path = ROOT / "query.txt"

gallery_sets = {"nm-01", "nm-02", "nm-03", "nm-04"}
query_sets = {"nm-05", "nm-06", "bg-01", "bg-02", "cl-01", "cl-02"}

train_lines = []
gallery_lines = []
query_lines = []

with all_path.open() as f:
    for line in f:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        seq_path, person_id = parts[0], parts[1]
        pid_num = int(person_id)
        seq_type = seq_path.split("/")[2]

        if 1 <= pid_num <= 74:
            train_lines.append(stripped)
        else:
            if seq_type in gallery_sets:
                gallery_lines.append(stripped)
            elif seq_type in query_sets:
                query_lines.append(stripped)

train_path.write_text("\n".join(train_lines) + "\n")
gallery_path.write_text("\n".join(gallery_lines) + "\n")
query_path.write_text("\n".join(query_lines) + "\n")
