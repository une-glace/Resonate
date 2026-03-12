"""
Step 3: Calculate mean AES metrics (CE/CU/PC/PQ) per system and dimension.
"""
import os
import json

def summarize_aes_scores():
    """Aggregate mean AES metrics for each system x dimension."""
    
    import sys
    result_jsonl = sys.argv[1]
    outfile = sys.argv[2]

    with open(outfile, 'w', encoding='utf-8') as f_out:
        if not os.path.exists(result_jsonl):
            print(f"[Warn] Missing: {result_jsonl}, skip.")

        total_ce = total_cu = total_pc = total_pq = count = 0

        with open(result_jsonl, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                if not line.startswith("{"):
                    continue
                data = json.loads(line)
                total_ce += data.get('CE', 0)
                total_cu += data.get('CU', 0)
                total_pc += data.get('PC', 0)
                total_pq += data.get('PQ', 0)
                count += 1

        if count > 0:
            avg_ce = total_ce / count
            avg_cu = total_cu / count
            avg_pc = total_pc / count
            avg_pq = total_pq / count
        else:
            avg_ce = avg_cu = avg_pc = avg_pq = 0

        print(f"count: {count}")
        print(f"Average CE: {avg_ce:.4f}")
        print(f"Average CU: {avg_cu:.4f}")
        print(f"Average PC: {avg_pc:.4f}")
        print(f"Average PQ: {avg_pq:.4f}")

        f_out.write(f"{avg_ce},{avg_cu},{avg_pc},{avg_pq}\n")

    print(f"\nSummary saved to {outfile}")

if __name__ == "__main__":
    summarize_aes_scores()
