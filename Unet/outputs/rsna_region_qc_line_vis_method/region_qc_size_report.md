# RSNA 4領域分割 面積QCレポート（rsna_line_vis同一方式）

## 方法
- 入力JSON: `Unet/outputs/rsna_line_predictions`
- JSONの `endpoints` は使わず、`centroid + angle_deg + 学習データline別平均線長` から短い線分を復元
- この復元方式は `Unet/outputs/rsna_line_vis/1.2.826.0.1.3680043.10001/C3_plane06.png` の4 Regionsパネルとピクセル完全一致確認済み
- 平均線長: line_1=38.7px, line_2=39.8px, line_3=39.2px, line_4=40.4px

## 要点
- 評価成功: 14050 vertebra planes
- 読み込み/生成エラー: 34
- 明確な面積異常候補: 55 (0.4%)
- C1 body near-zero: 0 / 2008。C1では頻発するため重症フラグから除外。

## 椎骨別 面積比中央値
| vertebra | n | outliers | body | right | left | posterior |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 2008 | 9 | 0.245 | 0.196 | 0.195 | 0.364 |
| C2 | 2011 | 7 | 0.209 | 0.162 | 0.157 | 0.471 |
| C3 | 2010 | 9 | 0.279 | 0.112 | 0.108 | 0.501 |
| C4 | 2010 | 4 | 0.294 | 0.109 | 0.105 | 0.492 |
| C5 | 2008 | 6 | 0.309 | 0.114 | 0.109 | 0.465 |
| C6 | 2005 | 6 | 0.317 | 0.111 | 0.106 | 0.459 |
| C7 | 1998 | 14 | 0.296 | 0.111 | 0.114 | 0.474 |

## フラグ種別
- foramen_severe_imbalance: 11
- right_foramen_near_zero: 6
- left_foramen_near_zero: 5
- posterior_dominant: 4
- body_near_zero: 3
- left_foramen_extreme_z23.5: 2
- left_foramen_extreme_z50.1: 2
- right_foramen_extreme_z53.9: 2
- left_foramen_extreme_z15.0: 2
- left_foramen_extreme_z16.3: 2
- posterior_extreme_z10.4: 2
- right_foramen_extreme_z8.7: 2
- left_foramen_extreme_z10.1: 2
- left_foramen_extreme_z8.4: 2
- right_foramen_extreme_z122.9: 1
- body_extreme_z9.0: 1
- right_foramen_extreme_z22.5: 1
- left_foramen_extreme_z117.9: 1
- posterior_extreme_z20.0: 1
- right_foramen_extreme_z96.3: 1
- posterior_extreme_z12.7: 1
- right_foramen_extreme_z80.5: 1
- body_extreme_z72.0: 1
- right_foramen_extreme_z76.5: 1
- left_foramen_extreme_z8.8: 1
- body_extreme_z71.1: 1
- right_foramen_extreme_z17.9: 1
- posterior_extreme_z8.1: 1
- body_extreme_z10.2: 1
- posterior_extreme_z10.2: 1
- body_extreme_z27.8: 1
- posterior_extreme_z16.0: 1
- right_foramen_extreme_z26.8: 1
- left_foramen_extreme_z19.4: 1
- left_foramen_extreme_z17.7: 1
- posterior_extreme_z14.4: 1
- left_foramen_extreme_z17.3: 1
- right_foramen_extreme_z16.6: 1
- right_foramen_extreme_z15.4: 1
- body_extreme_z9.9: 1
- posterior_extreme_z14.9: 1
- body_extreme_z14.9: 1
- left_foramen_extreme_z13.9: 1
- posterior_extreme_z12.0: 1
- posterior_extreme_z13.2: 1
- left_foramen_extreme_z12.8: 1
- body_extreme_z12.2: 1
- left_foramen_extreme_z10.8: 1
- right_foramen_extreme_z11.8: 1
- left_foramen_extreme_z11.6: 1
- posterior_extreme_z8.3: 1
- left_foramen_extreme_z11.5: 1
- body_extreme_z11.0: 1
- left_foramen_extreme_z9.1: 1
- right_foramen_extreme_z10.9: 1
- right_foramen_extreme_z10.8: 1
- body_extreme_z10.4: 1
- posterior_extreme_z10.0: 1
- left_foramen_extreme_z9.9: 1
- left_foramen_extreme_z9.8: 1
- right_foramen_extreme_z9.5: 1
- left_foramen_extreme_z9.2: 1
- body_extreme_z9.2: 1
- right_foramen_extreme_z8.2: 1
- posterior_extreme_z9.0: 1
- left_foramen_extreme_z9.0: 1
- right_foramen_extreme_z8.8: 1
- posterior_extreme_z8.6: 1
- left_foramen_extreme_z8.6: 1
- right_foramen_extreme_z8.4: 1
- posterior_extreme_z8.2: 1
- body_extreme_z8.1: 1

## 上位50件
| severity | study_uid | vertebra | plane | ratios B/R/L/P | flags |
|---:|---|---|---:|---|---|
| 122.9 | 1.2.826.0.1.3680043.24673 | C1 | 8 | 0.235/0.000/0.222/0.542 | right_foramen_near_zero:0.0000;right_foramen_extreme_z122.9:0.0000;foramen_severe_imbalance:0.0000 |
| 117.9 | 1.2.826.0.1.3680043.21283 | C1 | 8 | 0.091/0.025/0.000/0.884 | body_extreme_z9.0:0.0915;right_foramen_extreme_z22.5:0.0247;left_foramen_near_zero:0.0000;left_foramen_extreme_z117.9:0.0000;posterior_dominant:0.8838;posterior_extreme_z20.0:0.8838;foramen_severe_imbalance:0.0000 |
| 96.3 | 1.2.826.0.1.3680043.24673 | C2 | 10 | 0.167/0.000/0.010/0.823 | right_foramen_near_zero:0.0000;right_foramen_extreme_z96.3:0.0000;left_foramen_extreme_z23.5:0.0100;posterior_dominant:0.8227;posterior_extreme_z12.7:0.8227;foramen_severe_imbalance:0.0000 |
| 80.5 | 1.2.826.0.1.3680043.5541 | C3 | 6 | 0.575/0.000/0.160/0.265 | right_foramen_near_zero:0.0000;right_foramen_extreme_z80.5:0.0000;foramen_severe_imbalance:0.0000 |
| 80.0 | 1.2.826.0.1.3680043.14833 | C7 | 10 | 0.000/0.246/0.266/0.488 | body_near_zero:0.0000;body_extreme_z72.0:0.0000 |
| 80.0 | 1.2.826.0.1.3680043.21283 | C5 | 7 | 0.382/0.000/0.332/0.285 | right_foramen_near_zero:0.0000;right_foramen_extreme_z76.5:0.0000;left_foramen_extreme_z8.8:0.3321;foramen_severe_imbalance:0.0000 |
| 80.0 | 1.2.826.0.1.3680043.21283 | C6 | 6 | 0.104/0.147/0.000/0.750 | left_foramen_near_zero:0.0000;left_foramen_extreme_z50.1:0.0000;foramen_severe_imbalance:0.0000 |
| 80.0 | 1.2.826.0.1.3680043.25071 | C5 | 10 | 0.000/0.668/0.153/0.179 | body_near_zero:0.0000;body_extreme_z71.1:0.0000;right_foramen_extreme_z17.9:0.6679;posterior_extreme_z8.1:0.1791 |
| 80.0 | 1.2.826.0.1.3680043.28215 | C7 | 2 | 0.469/0.190/0.001/0.339 | left_foramen_near_zero:0.0013;left_foramen_extreme_z23.5:0.0013;foramen_severe_imbalance:0.0066 |
| 80.0 | 1.2.826.0.1.3680043.28657 | C7 | 5 | 0.063/0.000/0.107/0.830 | body_extreme_z10.2:0.0628;right_foramen_near_zero:0.0000;right_foramen_extreme_z53.9:0.0000;posterior_dominant:0.8299;posterior_extreme_z10.2:0.8299;foramen_severe_imbalance:0.0000 |
| 80.0 | 1.2.826.0.1.3680043.29045 | C6 | 5 | 0.402/0.121/0.000/0.477 | left_foramen_near_zero:0.0000;left_foramen_extreme_z50.1:0.0000;foramen_severe_imbalance:0.0000 |
| 80.0 | 1.2.826.0.1.3680043.4561 | C7 | 8 | 0.346/0.000/0.359/0.295 | right_foramen_near_zero:0.0000;right_foramen_extreme_z53.9:0.0000;foramen_severe_imbalance:0.0000 |
| 80.0 | 1.2.826.0.1.3680043.5055 | C6 | 8 | 0.423/0.193/0.004/0.380 | left_foramen_near_zero:0.0036;left_foramen_extreme_z15.0:0.0036;foramen_severe_imbalance:0.0187 |
| 80.0 | 1.2.826.0.1.3680043.8990 | C6 | 8 | 0.001/0.021/0.020/0.958 | body_near_zero:0.0014;body_extreme_z27.8:0.0014;posterior_dominant:0.9576;posterior_extreme_z16.0:0.9576 |
| 26.8 | 1.2.826.0.1.3680043.29791 | C1 | 7 | 0.420/0.016/0.199/0.365 | right_foramen_extreme_z26.8:0.0161 |
| 19.4 | 1.2.826.0.1.3680043.7831 | C1 | 6 | 0.332/0.249/0.030/0.388 | left_foramen_extreme_z19.4:0.0304 |
| 17.7 | 1.2.826.0.1.3680043.24673 | C3 | 7 | 0.162/0.136/0.619/0.083 | left_foramen_extreme_z17.7:0.6191;posterior_extreme_z14.4:0.0834 |
| 17.3 | 1.2.826.0.1.3680043.32023 | C1 | 7 | 0.431/0.219/0.038/0.312 | left_foramen_extreme_z17.3:0.0379 |
| 16.6 | 1.2.826.0.1.3680043.8362 | C3 | 8 | 0.368/0.011/0.093/0.528 | right_foramen_extreme_z16.6:0.0111 |
| 16.3 | 1.2.826.0.1.3680043.30864 | C5 | 9 | 0.411/0.120/0.009/0.460 | left_foramen_extreme_z16.3:0.0088 |
| 16.3 | 1.2.826.0.1.3680043.12109 | C2 | 8 | 0.211/0.576/0.024/0.188 | right_foramen_extreme_z15.4:0.5763;left_foramen_extreme_z16.3:0.0242;posterior_extreme_z10.4:0.1883 |
| 15.0 | 1.2.826.0.1.3680043.15680 | C4 | 8 | 0.692/0.093/0.011/0.204 | body_extreme_z9.9:0.6919;left_foramen_extreme_z15.0:0.0107 |
| 14.9 | 1.2.826.0.1.3680043.24673 | C7 | 8 | 0.202/0.413/0.313/0.071 | posterior_extreme_z14.9:0.0709 |
| 14.9 | 1.2.826.0.1.3680043.1429 | C7 | 8 | 0.028/0.335/0.090/0.547 | body_extreme_z14.9:0.0282 |
| 13.9 | 1.2.826.0.1.3680043.16729 | C7 | 7 | 0.138/0.082/0.669/0.110 | left_foramen_extreme_z13.9:0.6694;posterior_extreme_z12.0:0.1098 |
| 13.2 | 1.2.826.0.1.3680043.24673 | C4 | 7 | 0.515/0.126/0.272/0.086 | posterior_extreme_z13.2:0.0861 |
| 12.8 | 1.2.826.0.1.3680043.8362 | C7 | 4 | 0.568/0.061/0.010/0.361 | left_foramen_extreme_z12.8:0.0102 |
| 12.2 | 1.2.826.0.1.3680043.16729 | C4 | 6 | 0.050/0.249/0.390/0.311 | body_extreme_z12.2:0.0501;left_foramen_extreme_z10.8:0.3905 |
| 11.8 | 1.2.826.0.1.3680043.15680 | C2 | 5 | 0.330/0.042/0.110/0.518 | right_foramen_extreme_z11.8:0.0416 |
| 11.6 | 1.2.826.0.1.3680043.16729 | C5 | 7 | 0.273/0.116/0.439/0.172 | left_foramen_extreme_z11.6:0.4393;posterior_extreme_z8.3:0.1725 |
| 11.5 | 1.2.826.0.1.3680043.28215 | C6 | 11 | 0.249/0.203/0.008/0.540 | left_foramen_extreme_z11.5:0.0080 |
| 11.0 | 1.2.826.0.1.3680043.7221 | C1 | 9 | 0.072/0.228/0.386/0.314 | body_extreme_z11.0:0.0724;left_foramen_extreme_z9.1:0.3855 |
| 10.9 | 1.2.826.0.1.3680043.18945 | C3 | 6 | 0.347/0.025/0.130/0.498 | right_foramen_extreme_z10.9:0.0252 |
| 10.8 | 1.2.826.0.1.3680043.1750 | C1 | 6 | 0.344/0.076/0.227/0.353 | right_foramen_extreme_z10.8:0.0762 |
| 10.4 | 1.2.826.0.1.3680043.30051 | C7 | 4 | 0.060/0.456/0.039/0.445 | body_extreme_z10.4:0.0602;right_foramen_extreme_z8.7:0.4562 |
| 10.4 | 1.2.826.0.1.3680043.24673 | C5 | 7 | 0.519/0.133/0.221/0.127 | posterior_extreme_z10.4:0.1273 |
| 10.1 | 1.2.826.0.1.3680043.16729 | C1 | 8 | 0.194/0.354/0.077/0.375 | left_foramen_extreme_z10.1:0.0773 |
| 10.1 | 1.2.826.0.1.3680043.24562 | C2 | 9 | 0.329/0.148/0.051/0.472 | left_foramen_extreme_z10.1:0.0506 |
| 10.0 | 1.2.826.0.1.3680043.23978 | C7 | 3 | 0.196/0.365/0.293/0.146 | posterior_extreme_z10.0:0.1459 |
| 9.9 | 1.2.826.0.1.3680043.25172 | C2 | 7 | 0.254/0.146/0.052/0.549 | left_foramen_extreme_z9.9:0.0516 |
| 9.8 | 1.2.826.0.1.3680043.14654 | C1 | 8 | 0.349/0.261/0.080/0.310 | left_foramen_extreme_z9.8:0.0797 |
| 9.5 | 1.2.826.0.1.3680043.15680 | C3 | 7 | 0.352/0.031/0.155/0.463 | right_foramen_extreme_z9.5:0.0307 |
| 9.2 | 1.2.826.0.1.3680043.27996 | C2 | 6 | 0.285/0.169/0.056/0.489 | left_foramen_extreme_z9.2:0.0560 |
| 9.2 | 1.2.826.0.1.3680043.26389 | C7 | 8 | 0.075/0.333/0.117/0.475 | body_extreme_z9.2:0.0747 |
| 9.0 | 1.2.826.0.1.3680043.30864 | C7 | 8 | 0.131/0.427/0.274/0.168 | right_foramen_extreme_z8.2:0.4270;posterior_extreme_z9.0:0.1680 |
| 9.0 | 1.2.826.0.1.3680043.8368 | C3 | 8 | 0.451/0.144/0.032/0.374 | left_foramen_extreme_z9.0:0.0316 |
| 8.8 | 1.2.826.0.1.3680043.22006 | C7 | 7 | 0.459/0.018/0.226/0.297 | right_foramen_extreme_z8.8:0.0182 |
| 8.7 | 1.2.826.0.1.3680043.19388 | C7 | 8 | 0.437/0.018/0.121/0.424 | right_foramen_extreme_z8.7:0.0184 |
| 8.6 | 1.2.826.0.1.3680043.12253 | C5 | 5 | 0.557/0.157/0.120/0.166 | posterior_extreme_z8.6:0.1655 |
| 8.6 | 1.2.826.0.1.3680043.21283 | C4 | 6 | 0.152/0.110/0.312/0.426 | left_foramen_extreme_z8.6:0.3122 |

## 出力
- 全件CSV: `Unet/outputs/rsna_region_qc_line_vis_method/region_qc_details.csv`
- 面積異常候補CSV: `Unet/outputs/rsna_region_qc_line_vis_method/region_qc_size_outliers.csv`
- JSON: `Unet/outputs/rsna_region_qc_line_vis_method/region_qc_top_outliers.json`
- 上位可視化: `Unet/outputs/rsna_region_qc_line_vis_method/size_outlier_viz`
