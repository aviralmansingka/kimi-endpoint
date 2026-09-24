Window: 2026-08-25T20:10:00+00:00 — 2026-09-24T20:10:00+00:00
Analyzed 316 sessions; codex=117, claude=1, pi=198
| Metric | n | p50 | mean | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| prefix_text_system_plus_first_user | 316 | 2,568 | 5,986 | 13,772 | 22,086 | 25,302 |
| initial_recorded_text_context | 316 | 3,195 | 7,581 | 21,235 | 24,898 | 32,668 |
| first_request_input_real | 315 | 10,664 | 11,642 | 22,264 | 36,412 | 217,628 |
| turns_per_session | 316 | 12 | 36 | 98 | 297 | 624 |
| new_input_estimated | 11019 | 356 | 1,266 | 2,983 | 13,436 | 106,416 |
| new_input_excluding_args_estimated | 11019 | 150 | 1,085 | 2,698 | 13,238 | 106,395 |
| tool_output_per_turn_estimated | 11019 | 129 | 1,045 | 2,511 | 13,211 | 106,395 |
| output_real | 11334 | 243 | 537 | 1,201 | 4,894 | 20,223 |
| output_real_or_estimated | 11335 | 243 | 539 | 1,203 | 4,897 | 24,690 |
| assistant_text_only_estimated | 11335 | 8 | 50 | 94 | 576 | 24,671 |
| input_per_request_real | 11334 | 81,104 | 104,197 | 215,180 | 417,618 | 464,093 |
| total_session_input_real | 315 | 331,428 | 3,749,105 | 9,556,977 | 58,145,816 | 164,564,118 |
| total_session_output_real | 315 | 5,695 | 19,315 | 50,354 | 190,611 | 311,144 |
| cached_fraction_per_turn | 11334 | 0.989 | 0.912 | 0.998 | 0.999 | 1.000 |
| reasoning_fraction_per_turn | 11334 | 0.196 | 0.293 | 0.783 | 0.952 | 0.995 |
| parallel_tool_batch_width | 1774 | 3 | 3 | 4 | 7 | 10 |
| subagent_calls_per_session_positive | 27 | 2 | 3 | 6 | 7 | 7 |
| subagent_calls_per_response_positive | 59 | 1 | 1 | 2 | 3 | 3 |
