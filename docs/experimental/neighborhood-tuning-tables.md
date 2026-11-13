# Matched H100 neighborhood tuning tables

Resident complete-call CUDA graph medians in microseconds. Each cell uses
retained paired samples and requires independent accuracy, changed-input
graph validation and an actual kernel trace. These are operator measurements.
Eager results, ranges, samples, source hashes and all component errors remain
in the accompanying JSON. Screen samples select candidates; full and held-out
measurements evaluate those selections separately.

## initial-screen

| case | variant | forward | forward + backward | training range |
| --- | --- | ---: | ---: | ---: |
| 64sq-w7-bf16 | baseline | 36.96 | 153.00 | 152.88–153.10 |
| 64sq-w7-bf16 | no_stat_fills | 36.92 | 149.60 | 149.53–149.63 |
| 64sq-w7-bf16 | forward64 | 53.55 | 168.74 | 168.67–168.92 |
| 64sq-w7-bf16 | query64 | 36.92 | 159.55 | 159.35–159.84 |
| 64sq-w7-bf16 | keyvalue64 | 36.96 | 128.78 | 128.75–128.82 |
| 64sq-w7-bf16 | columns64 | 53.54 | 152.64 | 152.59–152.82 |
| 64sq-w7-bf16 | columns64_no_stat_fills | 53.47 | 150.21 | 150.19–150.24 |
| 64sq-w7-bf16 | flex | 32.40 | 122.51 | 122.49–122.59 |
| 64sq-w13-bf16 | baseline | 64.88 | 266.64 | 266.22–277.17 |
| 64sq-w13-bf16 | no_stat_fills | 64.80 | 273.01 | 266.15–273.29 |
| 64sq-w13-bf16 | forward64 | 94.51 | 307.07 | 296.58–307.97 |
| 64sq-w13-bf16 | query64 | 64.97 | 288.86 | 278.78–289.77 |
| 64sq-w13-bf16 | keyvalue64 | 64.89 | 226.72 | 226.62–235.58 |
| 64sq-w13-bf16 | columns64 | 94.53 | 271.07 | 268.07–275.54 |
| 64sq-w13-bf16 | columns64_no_stat_fills | 94.45 | 267.83 | 264.88–272.62 |
| 64sq-w13-bf16 | flex | 43.68 | 170.77 | 170.41–177.63 |
| tail-33x49-w7-bf16 | baseline | 24.80 | 85.25 | 85.08–85.31 |
| tail-33x49-w7-bf16 | no_stat_fills | 24.80 | 82.84 | 82.66–82.94 |
| tail-33x49-w7-bf16 | forward64 | 22.58 | 82.60 | 82.41–82.98 |
| tail-33x49-w7-bf16 | query64 | 24.78 | 87.70 | 87.65–88.30 |
| tail-33x49-w7-bf16 | keyvalue64 | 24.79 | 80.85 | 80.05–80.86 |
| tail-33x49-w7-bf16 | columns64 | 22.58 | 81.32 | 80.83–81.35 |
| tail-33x49-w7-bf16 | columns64_no_stat_fills | 22.61 | 78.48 | 78.35–79.03 |
| tail-33x49-w7-bf16 | flex | 23.60 | 79.21 | 79.10–79.25 |

## row-traversal-screen

| case | variant | forward | forward + backward | training range |
| --- | --- | ---: | ---: | ---: |
| 64sq-w7-bf16 | baseline | 36.97 | 153.13 | 153.03–153.20 |
| 64sq-w7-bf16 | no_stat_fills | 36.88 | 149.75 | 149.62–150.00 |
| 64sq-w7-bf16 | keyvalue64_no_stat_fills | 36.95 | 126.41 | 126.30–126.41 |
| 64sq-w7-bf16 | keyvalue128_no_stat_fills | 36.90 | 159.85 | 159.79–159.97 |
| 64sq-w7-bf16 | flex | 32.38 | 121.51 | 121.39–121.55 |
| 64sq-w7-bf16 | exact32_no_stat_fills | 34.39 | 147.83 | 147.57–148.04 |
| 64sq-w7-bf16 | exactmixed_no_stat_fills | 34.32 | 120.72 | 120.62–120.86 |
| 64sq-w7-bf16 | exactmixed128_no_stat_fills | 34.39 | 184.69 | 184.63–184.72 |
| 64sq-w13-bf16 | baseline | 64.89 | 271.54 | 269.44–276.65 |
| 64sq-w13-bf16 | no_stat_fills | 64.79 | 270.11 | 265.58–273.79 |
| 64sq-w13-bf16 | keyvalue64_no_stat_fills | 64.87 | 232.20 | 225.00–232.67 |
| 64sq-w13-bf16 | keyvalue128_no_stat_fills | 64.79 | 291.00 | 286.61–294.41 |
| 64sq-w13-bf16 | flex | 43.73 | 175.44 | 171.70–177.23 |
| 64sq-w13-bf16 | exact32_no_stat_fills | 60.13 | 257.90 | 257.15–266.82 |
| 64sq-w13-bf16 | exactmixed_no_stat_fills | 60.16 | 217.68 | 213.47–219.53 |
| 64sq-w13-bf16 | exactmixed128_no_stat_fills | 60.21 | 291.85 | 288.16–292.33 |
| tail-33x49-w7-bf16 | baseline | 25.02 | 85.06 | 85.06–85.22 |
| tail-33x49-w7-bf16 | no_stat_fills | 25.03 | 82.78 | 82.76–82.89 |
| tail-33x49-w7-bf16 | keyvalue64_no_stat_fills | 25.06 | 78.50 | 77.66–79.17 |
| tail-33x49-w7-bf16 | keyvalue128_no_stat_fills | 25.04 | 104.97 | 104.90–106.06 |
| tail-33x49-w7-bf16 | flex | 23.66 | 79.13 | 79.05–79.16 |
| tail-33x49-w7-bf16 | exact32_no_stat_fills | 25.06 | 82.98 | 82.26–83.28 |
| tail-33x49-w7-bf16 | exactmixed_no_stat_fills | 25.03 | 78.55 | 77.79–79.20 |
| tail-33x49-w7-bf16 | exactmixed128_no_stat_fills | 25.08 | 105.15 | 104.77–105.79 |

## query-restoration-screen

| case | variant | forward | forward + backward | training range |
| --- | --- | ---: | ---: | ---: |
| 64sq-w7-bf16 | baseline | 36.97 | 153.11 | 152.82–153.35 |
| 64sq-w7-bf16 | keyvalue64_no_stat_fills | 36.89 | 126.14 | 125.95–126.20 |
| 64sq-w7-bf16 | flex | 32.24 | 121.40 | 121.36–121.53 |
| 64sq-w7-bf16 | exactmixed_no_stat_fills | 34.32 | 120.56 | 120.32–120.67 |
| 64sq-w7-bf16 | exactmixed_noq_no_stat_fills | 34.35 | 116.73 | 116.50–117.04 |
| 64sq-w13-bf16 | baseline | 64.78 | 266.68 | 265.87–269.69 |
| 64sq-w13-bf16 | keyvalue64_no_stat_fills | 64.74 | 231.97 | 223.71–232.48 |
| 64sq-w13-bf16 | flex | 43.56 | 175.70 | 170.51–176.28 |
| 64sq-w13-bf16 | exactmixed_no_stat_fills | 60.11 | 213.76 | 211.56–219.38 |
| 64sq-w13-bf16 | exactmixed_noq_no_stat_fills | 60.04 | 206.24 | 205.91–213.44 |
| tail-33x49-w7-bf16 | baseline | 25.11 | 85.28 | 85.08–85.30 |
| tail-33x49-w7-bf16 | keyvalue64_no_stat_fills | 25.11 | 78.37 | 77.27–78.47 |
| tail-33x49-w7-bf16 | flex | 23.67 | 79.22 | 79.11–79.23 |
| tail-33x49-w7-bf16 | exactmixed_no_stat_fills | 25.13 | 79.18 | 78.51–79.21 |
| tail-33x49-w7-bf16 | exactmixed_noq_no_stat_fills | 25.05 | 79.20 | 78.47–79.21 |

## query-restoration-full

| case | variant | forward | forward + backward | training range |
| --- | --- | ---: | ---: | ---: |
| 32sq-w7-fp16 | baseline | 11.83 | 54.40 | 54.40–54.44 |
| 32sq-w7-fp16 | keyvalue64_no_stat_fills | 11.80 | 41.28 | 41.28–41.29 |
| 32sq-w7-fp16 | flex | 8.56 | 27.68 | 27.68–27.69 |
| 32sq-w7-fp16 | exactmixed_noq_no_stat_fills | 10.85 | 44.04 | 44.03–44.04 |
| 64sq-w7-fp16 | baseline | 37.34 | 167.55 | 165.99–169.26 |
| 64sq-w7-fp16 | keyvalue64_no_stat_fills | 37.30 | 125.74 | 123.63–126.80 |
| 64sq-w7-fp16 | flex | 32.26 | 121.83 | 121.53–122.51 |
| 64sq-w7-fp16 | exactmixed_noq_no_stat_fills | 33.34 | 116.83 | 116.73–120.54 |
| 64sq-w13-fp16 | baseline | 65.69 | 293.83 | 290.49–296.04 |
| 64sq-w13-fp16 | keyvalue64_no_stat_fills | 65.66 | 226.19 | 218.50–227.17 |
| 64sq-w13-fp16 | flex | 43.66 | 171.86 | 171.06–173.71 |
| 64sq-w13-fp16 | exactmixed_noq_no_stat_fills | 58.14 | 203.75 | 203.54–211.75 |
| 128sq-w7-fp16 | baseline | 222.38 | 700.79 | 695.99–706.52 |
| 128sq-w7-fp16 | keyvalue64_no_stat_fills | 222.24 | 769.88 | 757.04–777.87 |
| 128sq-w7-fp16 | flex | 159.23 | 600.94 | 596.62–604.79 |
| 128sq-w7-fp16 | exactmixed_noq_no_stat_fills | 222.15 | 762.07 | 757.26–777.02 |
| tail-33x49-w7-fp16 | baseline | 25.47 | 83.73 | 83.24–83.77 |
| tail-33x49-w7-fp16 | keyvalue64_no_stat_fills | 25.52 | 80.03 | 78.92–80.53 |
| tail-33x49-w7-fp16 | flex | 23.66 | 79.17 | 79.13–79.25 |
| tail-33x49-w7-fp16 | exactmixed_noq_no_stat_fills | 25.50 | 79.16 | 78.55–80.61 |
| 32sq-w7-bf16 | baseline | 11.75 | 49.08 | 49.07–49.08 |
| 32sq-w7-bf16 | keyvalue64_no_stat_fills | 11.78 | 41.85 | 41.85–41.86 |
| 32sq-w7-bf16 | flex | 8.55 | 27.63 | 27.63–27.64 |
| 32sq-w7-bf16 | exactmixed_noq_no_stat_fills | 11.41 | 45.77 | 45.77–45.78 |
| 64sq-w7-bf16 | baseline | 36.98 | 151.50 | 151.40–151.64 |
| 64sq-w7-bf16 | keyvalue64_no_stat_fills | 37.00 | 125.97 | 125.92–126.00 |
| 64sq-w7-bf16 | flex | 32.24 | 121.29 | 121.23–121.33 |
| 64sq-w7-bf16 | exactmixed_noq_no_stat_fills | 34.45 | 116.69 | 116.65–117.79 |
| 64sq-w13-bf16 | baseline | 64.77 | 270.56 | 265.99–274.03 |
| 64sq-w13-bf16 | keyvalue64_no_stat_fills | 64.82 | 230.90 | 223.67–231.84 |
| 64sq-w13-bf16 | flex | 43.68 | 171.53 | 170.55–172.36 |
| 64sq-w13-bf16 | exactmixed_noq_no_stat_fills | 60.14 | 205.03 | 204.80–211.07 |
| 128sq-w7-bf16 | baseline | 213.25 | 792.13 | 791.19–796.82 |
| 128sq-w7-bf16 | keyvalue64_no_stat_fills | 213.30 | 759.26 | 759.03–778.35 |
| 128sq-w7-bf16 | flex | 158.25 | 598.33 | 595.95–608.04 |
| 128sq-w7-bf16 | exactmixed_noq_no_stat_fills | 213.29 | 759.80 | 755.61–772.78 |
| tail-33x49-w7-bf16 | baseline | 24.81 | 85.56 | 85.03–85.79 |
| tail-33x49-w7-bf16 | keyvalue64_no_stat_fills | 24.77 | 78.71 | 78.47–79.17 |
| tail-33x49-w7-bf16 | flex | 23.64 | 79.19 | 79.14–79.78 |
| tail-33x49-w7-bf16 | exactmixed_noq_no_stat_fills | 24.77 | 78.20 | 77.42–79.25 |

## broad-default-full

| case | variant | forward | forward + backward | training range |
| --- | --- | ---: | ---: | ---: |
| 32sq-w7-fp16 | baseline | 11.83 | 54.06 | 54.04–54.06 |
| 32sq-w7-fp16 | keyvalue64_no_stat_fills | 11.79 | 41.13 | 41.13–41.14 |
| 32sq-w7-fp16 | flex | 8.55 | 27.71 | 27.70–27.71 |
| 32sq-w7-fp16 | production | 11.80 | 41.21 | 41.20–41.22 |
| 64sq-w7-fp16 | baseline | 37.37 | 168.51 | 166.04–169.80 |
| 64sq-w7-fp16 | keyvalue64_no_stat_fills | 37.33 | 125.38 | 123.62–126.39 |
| 64sq-w7-fp16 | flex | 32.42 | 121.77 | 121.45–122.37 |
| 64sq-w7-fp16 | production | 33.39 | 116.78 | 116.73–120.82 |
| 64sq-w13-fp16 | baseline | 65.67 | 290.79 | 288.36–298.55 |
| 64sq-w13-fp16 | keyvalue64_no_stat_fills | 65.64 | 226.81 | 218.57–227.26 |
| 64sq-w13-fp16 | flex | 43.68 | 171.06 | 169.97–173.86 |
| 64sq-w13-fp16 | production | 58.12 | 204.44 | 203.82–211.80 |
| 128sq-w7-fp16 | baseline | 221.07 | 701.29 | 697.90–708.91 |
| 128sq-w7-fp16 | keyvalue64_no_stat_fills | 220.94 | 773.08 | 761.18–774.40 |
| 128sq-w7-fp16 | flex | 158.37 | 601.38 | 600.01–608.61 |
| 128sq-w7-fp16 | production | 220.92 | 695.15 | 692.27–705.77 |
| tail-33x49-w7-fp16 | baseline | 25.54 | 83.62 | 83.34–83.73 |
| tail-33x49-w7-fp16 | keyvalue64_no_stat_fills | 25.59 | 79.04 | 78.81–80.20 |
| tail-33x49-w7-fp16 | flex | 23.70 | 79.20 | 79.16–79.31 |
| tail-33x49-w7-fp16 | production | 25.58 | 79.47 | 78.84–80.11 |
| 32sq-w7-bf16 | baseline | 11.73 | 49.15 | 49.14–49.15 |
| 32sq-w7-bf16 | keyvalue64_no_stat_fills | 11.78 | 41.80 | 41.79–41.80 |
| 32sq-w7-bf16 | flex | 8.54 | 27.66 | 27.66–27.66 |
| 32sq-w7-bf16 | production | 11.76 | 41.83 | 41.83–41.84 |
| 64sq-w7-bf16 | baseline | 37.00 | 152.28 | 152.16–152.34 |
| 64sq-w7-bf16 | keyvalue64_no_stat_fills | 37.02 | 125.93 | 125.89–126.23 |
| 64sq-w7-bf16 | flex | 32.44 | 121.33 | 121.25–121.44 |
| 64sq-w7-bf16 | production | 34.42 | 116.73 | 116.67–117.75 |
| 64sq-w13-bf16 | baseline | 64.80 | 270.04 | 266.41–272.31 |
| 64sq-w13-bf16 | keyvalue64_no_stat_fills | 64.85 | 228.20 | 223.49–229.47 |
| 64sq-w13-bf16 | flex | 43.63 | 171.32 | 170.77–172.13 |
| 64sq-w13-bf16 | production | 60.17 | 206.22 | 205.35–211.36 |
| 128sq-w7-bf16 | baseline | 213.34 | 793.27 | 788.70–801.11 |
| 128sq-w7-bf16 | keyvalue64_no_stat_fills | 213.34 | 759.91 | 759.42–772.83 |
| 128sq-w7-bf16 | flex | 158.39 | 599.68 | 596.64–614.01 |
| 128sq-w7-bf16 | production | 213.32 | 789.40 | 784.82–793.91 |
| tail-33x49-w7-bf16 | baseline | 25.09 | 85.01 | 85.00–85.10 |
| tail-33x49-w7-bf16 | keyvalue64_no_stat_fills | 25.05 | 78.43 | 77.63–79.15 |
| tail-33x49-w7-bf16 | flex | 23.68 | 79.13 | 79.10–79.25 |
| tail-33x49-w7-bf16 | production | 25.06 | 78.37 | 78.02–78.39 |

## broad-default-evaluation

| case | variant | forward | forward + backward | training range |
| --- | --- | ---: | ---: | ---: |
| eval-48x64-w9-fp16 | baseline | 37.60 | 124.58 | 123.52–124.69 |
| eval-48x64-w9-fp16 | keyvalue64_no_stat_fills | 37.53 | 133.43 | 133.31–134.09 |
| eval-48x64-w9-fp16 | flex | 27.47 | 101.05 | 100.92–101.52 |
| eval-48x64-w9-fp16 | production | 32.92 | 124.57 | 124.52–125.10 |
| eval-96x32-w5-fp16 | baseline | 15.47 | 52.27 | 52.23–52.42 |
| eval-96x32-w5-fp16 | keyvalue64_no_stat_fills | 15.43 | 57.60 | 57.54–57.64 |
| eval-96x32-w5-fp16 | flex | 18.07 | 62.06 | 62.01–62.08 |
| eval-96x32-w5-fp16 | production | 15.45 | 57.67 | 57.65–57.70 |
| eval-45x51-w9-fp16 | baseline | 39.98 | 131.77 | 131.71–131.79 |
| eval-45x51-w9-fp16 | keyvalue64_no_stat_fills | 39.95 | 123.17 | 123.07–123.19 |
| eval-45x51-w9-fp16 | flex | 37.26 | 110.35 | 110.30–110.43 |
| eval-45x51-w9-fp16 | production | 39.95 | 123.00 | 122.98–123.07 |
| eval-64sq-w7-h1-fp16 | baseline | 17.58 | 62.88 | 62.88–62.89 |
| eval-64sq-w7-h1-fp16 | keyvalue64_no_stat_fills | 17.54 | 57.51 | 57.50–57.51 |
| eval-64sq-w7-h1-fp16 | flex | 12.41 | 33.89 | 33.88–33.89 |
| eval-64sq-w7-h1-fp16 | production | 17.56 | 60.81 | 60.80–60.81 |
| eval-64sq-w7-h16-fp16 | baseline | 73.30 | 285.20 | 282.08–286.55 |
| eval-64sq-w7-h16-fp16 | keyvalue64_no_stat_fills | 73.27 | 245.54 | 240.69–247.59 |
| eval-64sq-w7-h16-fp16 | flex | 61.56 | 228.11 | 227.10–229.37 |
| eval-64sq-w7-h16-fp16 | production | 65.51 | 227.56 | 227.48–233.50 |
| eval-64x47-w11-fp16 | baseline | 44.90 | 161.65 | 161.58–161.73 |
| eval-64x47-w11-fp16 | keyvalue64_no_stat_fills | 44.94 | 149.96 | 149.87–150.06 |
| eval-64x47-w11-fp16 | flex | 35.74 | 123.50 | 123.40–123.68 |
| eval-64x47-w11-fp16 | production | 44.93 | 149.79 | 149.66–149.86 |
| eval-48x64-w9-bf16 | baseline | 37.33 | 167.73 | 167.60–170.15 |
| eval-48x64-w9-bf16 | keyvalue64_no_stat_fills | 37.35 | 134.04 | 133.97–137.71 |
| eval-48x64-w9-bf16 | flex | 27.43 | 101.64 | 101.59–101.90 |
| eval-48x64-w9-bf16 | production | 34.33 | 123.54 | 123.39–128.00 |
| eval-96x32-w5-bf16 | baseline | 15.38 | 68.51 | 68.47–68.53 |
| eval-96x32-w5-bf16 | keyvalue64_no_stat_fills | 15.42 | 57.01 | 56.96–57.03 |
| eval-96x32-w5-bf16 | flex | 18.09 | 61.58 | 61.56–61.59 |
| eval-96x32-w5-bf16 | production | 15.39 | 56.93 | 56.90–56.94 |
| eval-45x51-w9-bf16 | baseline | 40.49 | 127.23 | 125.98–130.22 |
| eval-45x51-w9-bf16 | keyvalue64_no_stat_fills | 40.52 | 123.55 | 123.49–123.62 |
| eval-45x51-w9-bf16 | flex | 37.22 | 108.18 | 108.15–108.23 |
| eval-45x51-w9-bf16 | production | 40.51 | 123.53 | 123.47–123.56 |
| eval-64sq-w7-h1-bf16 | baseline | 17.56 | 77.59 | 77.58–77.59 |
| eval-64sq-w7-h1-bf16 | keyvalue64_no_stat_fills | 17.52 | 57.94 | 57.94–57.95 |
| eval-64sq-w7-h1-bf16 | flex | 12.36 | 33.78 | 33.77–33.78 |
| eval-64sq-w7-h1-bf16 | production | 17.53 | 75.34 | 75.33–75.34 |
| eval-64sq-w7-h16-bf16 | baseline | 72.87 | 268.50 | 268.42–268.67 |
| eval-64sq-w7-h16-bf16 | keyvalue64_no_stat_fills | 72.78 | 245.41 | 245.38–245.45 |
| eval-64sq-w7-h16-bf16 | flex | 61.91 | 226.96 | 226.89–227.00 |
| eval-64sq-w7-h16-bf16 | production | 69.23 | 228.84 | 228.75–228.92 |
| eval-64x47-w11-bf16 | baseline | 45.25 | 147.88 | 147.18–149.42 |
| eval-64x47-w11-bf16 | keyvalue64_no_stat_fills | 45.23 | 152.61 | 152.48–152.72 |
| eval-64x47-w11-bf16 | flex | 35.57 | 123.97 | 123.91–124.99 |
| eval-64x47-w11-bf16 | production | 45.22 | 152.46 | 152.38–152.50 |

## final-default-matrix

| case | variant | forward | forward + backward | training range |
| --- | --- | ---: | ---: | ---: |
| 32sq-w7-fp16 | baseline | 11.86 | 54.47 | 54.46–54.49 |
| 32sq-w7-fp16 | keyvalue64_no_stat_fills | 11.78 | 41.28 | 41.26–41.28 |
| 32sq-w7-fp16 | flex | 8.54 | 27.70 | 27.70–27.70 |
| 32sq-w7-fp16 | production | 11.78 | 41.29 | 41.28–41.30 |
| 64sq-w7-fp16 | baseline | 37.35 | 167.26 | 165.69–168.94 |
| 64sq-w7-fp16 | keyvalue64_no_stat_fills | 37.31 | 125.67 | 123.38–126.94 |
| 64sq-w7-fp16 | flex | 32.28 | 122.27 | 121.64–123.15 |
| 64sq-w7-fp16 | production | 33.33 | 116.71 | 116.63–120.17 |
| 64sq-w13-fp16 | baseline | 65.70 | 290.69 | 288.75–296.00 |
| 64sq-w13-fp16 | keyvalue64_no_stat_fills | 65.67 | 224.70 | 218.93–225.86 |
| 64sq-w13-fp16 | flex | 43.70 | 172.25 | 170.96–174.33 |
| 64sq-w13-fp16 | production | 58.16 | 204.33 | 204.29–210.73 |
| 128sq-w7-fp16 | baseline | 220.97 | 698.74 | 696.80–704.01 |
| 128sq-w7-fp16 | keyvalue64_no_stat_fills | 220.89 | 771.47 | 756.79–774.68 |
| 128sq-w7-fp16 | flex | 158.28 | 597.86 | 596.75–608.80 |
| 128sq-w7-fp16 | production | 220.88 | 688.57 | 688.27–705.01 |
| tail-33x49-w7-fp16 | baseline | 25.39 | 83.48 | 83.40–83.75 |
| tail-33x49-w7-fp16 | keyvalue64_no_stat_fills | 25.40 | 79.36 | 78.91–80.04 |
| tail-33x49-w7-fp16 | flex | 23.56 | 79.88 | 79.87–79.95 |
| tail-33x49-w7-fp16 | production | 25.40 | 79.41 | 78.68–80.09 |
| 32sq-w7-bf16 | baseline | 11.76 | 49.12 | 49.12–49.15 |
| 32sq-w7-bf16 | keyvalue64_no_stat_fills | 11.80 | 41.80 | 41.80–41.82 |
| 32sq-w7-bf16 | flex | 8.52 | 27.72 | 27.72–27.73 |
| 32sq-w7-bf16 | production | 11.77 | 41.92 | 41.92–41.94 |
| 64sq-w7-bf16 | baseline | 36.96 | 151.82 | 151.78–151.84 |
| 64sq-w7-bf16 | keyvalue64_no_stat_fills | 36.98 | 126.09 | 126.05–126.13 |
| 64sq-w7-bf16 | flex | 32.16 | 121.51 | 121.46–121.56 |
| 64sq-w7-bf16 | production | 34.39 | 116.61 | 116.58–116.65 |
| 64sq-w13-bf16 | baseline | 64.72 | 269.54 | 266.24–274.21 |
| 64sq-w13-bf16 | keyvalue64_no_stat_fills | 64.76 | 230.34 | 223.51–231.76 |
| 64sq-w13-bf16 | flex | 43.59 | 171.66 | 170.85–173.74 |
| 64sq-w13-bf16 | production | 60.08 | 205.61 | 205.56–212.47 |
| 128sq-w7-bf16 | baseline | 213.46 | 790.11 | 789.57–808.95 |
| 128sq-w7-bf16 | keyvalue64_no_stat_fills | 213.47 | 758.15 | 756.68–776.29 |
| 128sq-w7-bf16 | flex | 158.26 | 596.82 | 596.75–601.76 |
| 128sq-w7-bf16 | production | 213.50 | 786.40 | 786.19–807.87 |
| tail-33x49-w7-bf16 | baseline | 24.68 | 85.12 | 85.11–85.13 |
| tail-33x49-w7-bf16 | keyvalue64_no_stat_fills | 24.64 | 78.40 | 78.07–78.46 |
| tail-33x49-w7-bf16 | flex | 23.65 | 79.78 | 79.72–79.93 |
| tail-33x49-w7-bf16 | production | 24.65 | 78.38 | 77.04–78.48 |

## fresh-final-evaluation

| case | variant | forward | forward + backward | training range |
| --- | --- | ---: | ---: | ---: |
| fresh-64x32-w5-fp16 | baseline | 12.79 | 53.28 | 52.72–53.35 |
| fresh-64x32-w5-fp16 | keyvalue64_no_stat_fills | 12.72 | 41.52 | 41.28–41.66 |
| fresh-64x32-w5-fp16 | flex | 13.45 | 44.43 | 44.42–44.44 |
| fresh-64x32-w5-fp16 | production | 12.72 | 41.65 | 41.57–41.92 |
| fresh-40x64-w5-fp16 | baseline | 22.51 | 74.08 | 74.04–76.12 |
| fresh-40x64-w5-fp16 | keyvalue64_no_stat_fills | 22.47 | 77.92 | 77.87–78.12 |
| fresh-40x64-w5-fp16 | flex | 17.68 | 59.74 | 59.67–59.79 |
| fresh-40x64-w5-fp16 | production | 22.49 | 72.13 | 71.55–73.23 |
| fresh-52x64-w5-fp16 | baseline | 27.68 | 94.03 | 93.96–94.05 |
| fresh-52x64-w5-fp16 | keyvalue64_no_stat_fills | 27.65 | 90.23 | 90.19–90.28 |
| fresh-52x64-w5-fp16 | flex | 20.89 | 72.45 | 72.43–72.48 |
| fresh-52x64-w5-fp16 | production | 24.84 | 95.65 | 95.63–95.68 |
| fresh-54x49-w11-fp16 | baseline | 41.95 | 157.11 | 156.98–157.23 |
| fresh-54x49-w11-fp16 | keyvalue64_no_stat_fills | 41.90 | 150.70 | 150.59–150.78 |
| fresh-54x49-w11-fp16 | flex | 34.50 | 117.26 | 117.15–117.46 |
| fresh-54x49-w11-fp16 | production | 41.93 | 154.31 | 154.19–154.62 |
| fresh-64x32-w5-bf16 | baseline | 12.69 | 49.15 | 49.03–49.25 |
| fresh-64x32-w5-bf16 | keyvalue64_no_stat_fills | 12.70 | 41.94 | 41.72–42.08 |
| fresh-64x32-w5-bf16 | flex | 13.36 | 44.36 | 44.34–44.36 |
| fresh-64x32-w5-bf16 | production | 12.70 | 41.96 | 41.72–42.00 |
| fresh-40x64-w5-bf16 | baseline | 22.43 | 97.45 | 97.40–97.49 |
| fresh-40x64-w5-bf16 | keyvalue64_no_stat_fills | 22.47 | 77.38 | 77.36–77.40 |
| fresh-40x64-w5-bf16 | flex | 17.65 | 59.19 | 59.17–59.21 |
| fresh-40x64-w5-bf16 | production | 22.45 | 94.88 | 94.83–94.91 |
| fresh-52x64-w5-bf16 | baseline | 27.44 | 110.24 | 110.22–110.27 |
| fresh-52x64-w5-bf16 | keyvalue64_no_stat_fills | 27.47 | 90.54 | 90.53–90.58 |
| fresh-52x64-w5-bf16 | flex | 20.93 | 71.48 | 71.47–71.49 |
| fresh-52x64-w5-bf16 | production | 25.64 | 99.00 | 98.95–99.06 |
| fresh-54x49-w11-bf16 | baseline | 43.37 | 161.84 | 161.75–161.96 |
| fresh-54x49-w11-bf16 | keyvalue64_no_stat_fills | 43.43 | 149.69 | 149.56–149.90 |
| fresh-54x49-w11-bf16 | flex | 34.46 | 117.37 | 117.27–117.40 |
| fresh-54x49-w11-bf16 | production | 43.42 | 159.54 | 159.38–159.67 |
