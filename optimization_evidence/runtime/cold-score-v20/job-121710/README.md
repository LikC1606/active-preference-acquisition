# Current v1.2.0 cold score

Job 121710 ran on nv-h100-022 with an 8-vCPU, 40 GiB VM and 0 GPU. The host allocation was 16 logical CPUs bound to 8 distinct physical cores. Dev and Hidden each trained the submitted best method from an empty candidate state for seeds 101, 202, 303, 404, and 505. Both phases passed within 7,200 seconds; the evidence contains the reward summaries, resource checks, receipts, and container inspections.

Dev: 5,196.04 seconds, score 0.24927529058448603. Hidden: 4,769.36 seconds, score 0.24433649204862387. The ten new training runs used no model API requests.
