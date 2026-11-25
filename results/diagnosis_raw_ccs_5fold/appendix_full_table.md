# Appendix: all models and diagnosis representations

| Model | Representation | PR_AUC_mean | AUROC_mean | Precision_mean | Recall_mean | F1_mean | Brier_mean | PR_AUC_std | AUROC_std | Precision_std | Recall_std | F1_std | Brier_std | Pooled_OOF_PR_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | Raw | 0.212744 | 0.656468 | 0.261128 | 0.229185 | 0.244116 | 0.097284 | 0.004608 | 0.003292 | 0.004115 | 0.003620 | 0.003852 | 0.000276 | 0.212099 |
| LR | CCS Replacement | 0.213274 | 0.660160 | 0.260926 | 0.229007 | 0.243927 | 0.097085 | 0.004099 | 0.003119 | 0.005532 | 0.004803 | 0.005142 | 0.000242 | 0.212677 |
| LR | Raw + CCS | 0.211677 | 0.654964 | 0.259517 | 0.227771 | 0.242610 | 0.097408 | 0.004642 | 0.003407 | 0.005167 | 0.004552 | 0.004840 | 0.000275 | 0.211108 |
| RF | Raw | 0.222174 | 0.670451 | 0.267976 | 0.235194 | 0.250517 | 0.098313 | 0.003724 | 0.005499 | 0.009389 | 0.008186 | 0.008746 | 0.000131 | 0.221440 |
| RF | CCS Replacement | 0.224593 | 0.672140 | 0.270292 | 0.237226 | 0.252682 | 0.097616 | 0.005393 | 0.004939 | 0.007981 | 0.006926 | 0.007416 | 0.000090 | 0.223811 |
| RF | Raw + CCS | 0.220492 | 0.669773 | 0.264753 | 0.232365 | 0.247504 | 0.098412 | 0.004306 | 0.004068 | 0.008961 | 0.007781 | 0.008330 | 0.000063 | 0.219881 |
| XGBoost | Raw | 0.228613 | 0.671367 | 0.270695 | 0.237582 | 0.253060 | 0.096189 | 0.003085 | 0.006056 | 0.007019 | 0.006159 | 0.006561 | 0.000438 | 0.226654 |
| XGBoost | CCS Replacement | 0.229900 | 0.673298 | 0.271803 | 0.238553 | 0.254094 | 0.095979 | 0.004635 | 0.005166 | 0.007656 | 0.006654 | 0.007120 | 0.000214 | 0.228867 |
| XGBoost | Raw + CCS | 0.232339 | 0.674912 | 0.275730 | 0.242000 | 0.257766 | 0.095847 | 0.002972 | 0.006636 | 0.007258 | 0.006306 | 0.006749 | 0.000175 | 0.231234 |
| HGB | Raw | 0.237247 | 0.676933 | 0.279557 | 0.245358 | 0.261343 | 0.095573 | 0.002473 | 0.005916 | 0.009157 | 0.007978 | 0.008527 | 0.000186 | 0.235987 |
| HGB | CCS Replacement | 0.235817 | 0.676216 | 0.279053 | 0.244915 | 0.260872 | 0.095651 | 0.004125 | 0.006044 | 0.010930 | 0.009507 | 0.010169 | 0.000230 | 0.235035 |
| HGB | Raw + CCS | 0.239179 | 0.678333 | 0.284491 | 0.249688 | 0.265956 | 0.095470 | 0.004208 | 0.006897 | 0.011619 | 0.010131 | 0.010824 | 0.000282 | 0.237823 |
| CatBoost | Raw | 0.246506 | 0.682839 | 0.285196 | 0.250308 | 0.266616 | 0.095084 | 0.004438 | 0.004716 | 0.004798 | 0.004132 | 0.004441 | 0.000200 | 0.245430 |
| CatBoost | CCS Replacement | 0.244229 | 0.679118 | 0.280866 | 0.246507 | 0.262567 | 0.095234 | 0.006686 | 0.004907 | 0.009965 | 0.008682 | 0.009280 | 0.000293 | 0.243098 |
| CatBoost | Raw + CCS | 0.246917 | 0.682729 | 0.285096 | 0.250219 | 0.266521 | 0.095059 | 0.005892 | 0.004983 | 0.008323 | 0.007223 | 0.007734 | 0.000264 | 0.245780 |

## Fold-wise results

| Model | Representation | Fold | PR_AUC | AUROC | Precision | Recall | F1_score | Brier_Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | Raw | 1 | 0.208819 | 0.651344 | 0.257805 | 0.226148 | 0.240941 | 0.097513 |
| LR | Raw | 2 | 0.216470 | 0.655958 | 0.256798 | 0.225464 | 0.240113 | 0.097255 |
| LR | Raw | 3 | 0.209879 | 0.660121 | 0.263343 | 0.231211 | 0.246234 | 0.097405 |
| LR | Raw | 4 | 0.218899 | 0.656580 | 0.266868 | 0.234202 | 0.249470 | 0.096819 |
| LR | Raw | 5 | 0.209651 | 0.658337 | 0.260826 | 0.228900 | 0.243822 | 0.097430 |
| LR | CCS Replacement | 1 | 0.213963 | 0.659882 | 0.263847 | 0.231449 | 0.246588 | 0.097019 |
| LR | CCS Replacement | 2 | 0.215956 | 0.657952 | 0.251762 | 0.221043 | 0.235405 | 0.097040 |
| LR | CCS Replacement | 3 | 0.207855 | 0.659989 | 0.261329 | 0.229443 | 0.244350 | 0.097472 |
| LR | CCS Replacement | 4 | 0.218044 | 0.657590 | 0.266365 | 0.233760 | 0.249000 | 0.096805 |
| LR | CCS Replacement | 5 | 0.210551 | 0.665387 | 0.261329 | 0.229342 | 0.244293 | 0.097089 |
| LR | Raw + CCS | 1 | 0.207264 | 0.649928 | 0.256798 | 0.225265 | 0.240000 | 0.097683 |
| LR | Raw + CCS | 2 | 0.215973 | 0.654544 | 0.255287 | 0.224138 | 0.238701 | 0.097342 |
| LR | Raw + CCS | 3 | 0.209778 | 0.659219 | 0.264350 | 0.232095 | 0.247175 | 0.097452 |
| LR | Raw + CCS | 4 | 0.217301 | 0.654551 | 0.265861 | 0.233319 | 0.248529 | 0.096975 |
| LR | Raw + CCS | 5 | 0.208068 | 0.656578 | 0.255287 | 0.224039 | 0.238644 | 0.097590 |
| RF | Raw | 1 | 0.219363 | 0.669692 | 0.269386 | 0.236307 | 0.251765 | 0.098509 |
| RF | Raw | 2 | 0.219848 | 0.661537 | 0.255791 | 0.224580 | 0.239171 | 0.098372 |
| RF | Raw | 3 | 0.220422 | 0.671361 | 0.261329 | 0.229443 | 0.244350 | 0.098287 |
| RF | Raw | 4 | 0.228391 | 0.673906 | 0.278449 | 0.244366 | 0.260297 | 0.098203 |
| RF | Raw | 5 | 0.222848 | 0.675759 | 0.274924 | 0.241273 | 0.257002 | 0.098194 |
| RF | CCS Replacement | 1 | 0.229388 | 0.671809 | 0.277442 | 0.243375 | 0.259294 | 0.097571 |
| RF | CCS Replacement | 2 | 0.219126 | 0.663774 | 0.259315 | 0.227675 | 0.242467 | 0.097744 |
| RF | CCS Replacement | 3 | 0.221342 | 0.673875 | 0.264350 | 0.232095 | 0.247175 | 0.097657 |
| RF | CCS Replacement | 4 | 0.231309 | 0.675947 | 0.274924 | 0.241273 | 0.257002 | 0.097505 |
| RF | CCS Replacement | 5 | 0.221799 | 0.675297 | 0.275428 | 0.241715 | 0.257472 | 0.097603 |
| RF | Raw + CCS | 1 | 0.223705 | 0.670100 | 0.275428 | 0.241608 | 0.257412 | 0.098405 |
| RF | Raw + CCS | 2 | 0.221734 | 0.663826 | 0.254783 | 0.223696 | 0.238230 | 0.098383 |
| RF | Raw + CCS | 3 | 0.212918 | 0.668683 | 0.256294 | 0.225022 | 0.239642 | 0.098511 |
| RF | Raw + CCS | 4 | 0.222242 | 0.671253 | 0.266868 | 0.234202 | 0.249470 | 0.098421 |
| RF | Raw + CCS | 5 | 0.221861 | 0.675001 | 0.270393 | 0.237296 | 0.252765 | 0.098341 |
| XGBoost | Raw | 1 | 0.223807 | 0.669422 | 0.264350 | 0.231890 | 0.247059 | 0.096924 |
| XGBoost | Raw | 2 | 0.228837 | 0.662570 | 0.265861 | 0.233422 | 0.248588 | 0.096224 |
| XGBoost | Raw | 3 | 0.228117 | 0.675389 | 0.268882 | 0.236074 | 0.251412 | 0.095968 |
| XGBoost | Raw | 4 | 0.230210 | 0.671066 | 0.272407 | 0.239063 | 0.254648 | 0.096029 |
| XGBoost | Raw | 5 | 0.232093 | 0.678386 | 0.281974 | 0.247459 | 0.263591 | 0.095801 |
| XGBoost | CCS Replacement | 1 | 0.228741 | 0.676801 | 0.274924 | 0.241166 | 0.256941 | 0.096021 |
| XGBoost | CCS Replacement | 2 | 0.227997 | 0.664967 | 0.264350 | 0.232095 | 0.247175 | 0.096146 |
| XGBoost | CCS Replacement | 3 | 0.226503 | 0.672154 | 0.262840 | 0.230769 | 0.245763 | 0.096159 |
| XGBoost | CCS Replacement | 4 | 0.238057 | 0.674564 | 0.278953 | 0.244808 | 0.260767 | 0.095635 |
| XGBoost | CCS Replacement | 5 | 0.228201 | 0.678005 | 0.277946 | 0.243924 | 0.259826 | 0.095934 |
| XGBoost | Raw + CCS | 1 | 0.230655 | 0.679132 | 0.279456 | 0.245141 | 0.261176 | 0.095888 |
| XGBoost | Raw + CCS | 2 | 0.230669 | 0.664109 | 0.264854 | 0.232538 | 0.247646 | 0.096078 |
| XGBoost | Raw + CCS | 3 | 0.229878 | 0.675649 | 0.272407 | 0.239169 | 0.254708 | 0.095921 |
| XGBoost | Raw + CCS | 4 | 0.237082 | 0.674370 | 0.278449 | 0.244366 | 0.260297 | 0.095708 |
| XGBoost | Raw + CCS | 5 | 0.233414 | 0.681298 | 0.283484 | 0.248785 | 0.265004 | 0.095639 |
| HGB | Raw | 1 | 0.235802 | 0.677889 | 0.284491 | 0.249558 | 0.265882 | 0.095651 |
| HGB | Raw | 2 | 0.235269 | 0.666667 | 0.265861 | 0.233422 | 0.248588 | 0.095854 |
| HGB | Raw | 3 | 0.237747 | 0.680836 | 0.277946 | 0.244032 | 0.259887 | 0.095485 |
| HGB | Raw | 4 | 0.241348 | 0.678252 | 0.290534 | 0.254971 | 0.271593 | 0.095374 |
| HGB | Raw | 5 | 0.236067 | 0.681022 | 0.278953 | 0.244808 | 0.260767 | 0.095501 |
| HGB | CCS Replacement | 1 | 0.236774 | 0.680819 | 0.292044 | 0.256184 | 0.272941 | 0.095517 |
| HGB | CCS Replacement | 2 | 0.229375 | 0.665895 | 0.269889 | 0.236958 | 0.252354 | 0.096040 |
| HGB | CCS Replacement | 3 | 0.236414 | 0.676271 | 0.266365 | 0.233864 | 0.249058 | 0.095639 |
| HGB | CCS Replacement | 4 | 0.240843 | 0.677967 | 0.287009 | 0.251878 | 0.268298 | 0.095452 |
| HGB | CCS Replacement | 5 | 0.235676 | 0.680130 | 0.279960 | 0.245692 | 0.261709 | 0.095606 |
| HGB | Raw + CCS | 1 | 0.240550 | 0.681268 | 0.293051 | 0.257067 | 0.273882 | 0.095405 |
| HGB | Raw + CCS | 2 | 0.232899 | 0.666464 | 0.266365 | 0.233864 | 0.249058 | 0.095949 |
| HGB | Raw + CCS | 3 | 0.239505 | 0.679114 | 0.282477 | 0.248011 | 0.264124 | 0.095439 |
| HGB | Raw + CCS | 4 | 0.244546 | 0.680544 | 0.296073 | 0.259832 | 0.276771 | 0.095211 |
| HGB | Raw + CCS | 5 | 0.238394 | 0.684273 | 0.284491 | 0.249669 | 0.265945 | 0.095345 |
| CatBoost | Raw | 1 | 0.250130 | 0.684952 | 0.288520 | 0.253092 | 0.269647 | 0.094877 |
| CatBoost | Raw | 2 | 0.241228 | 0.675526 | 0.280463 | 0.246242 | 0.262241 | 0.095386 |
| CatBoost | Raw | 3 | 0.244320 | 0.686355 | 0.279456 | 0.245358 | 0.261299 | 0.095095 |
| CatBoost | Raw | 4 | 0.252006 | 0.680745 | 0.288520 | 0.253204 | 0.269711 | 0.094931 |
| CatBoost | Raw | 5 | 0.244849 | 0.686618 | 0.289023 | 0.253646 | 0.270181 | 0.095130 |
| CatBoost | CCS Replacement | 1 | 0.249894 | 0.682631 | 0.283988 | 0.249117 | 0.265412 | 0.094917 |
| CatBoost | CCS Replacement | 2 | 0.236883 | 0.671147 | 0.268379 | 0.235632 | 0.250942 | 0.095639 |
| CatBoost | CCS Replacement | 3 | 0.239144 | 0.680575 | 0.272407 | 0.239169 | 0.254708 | 0.095382 |
| CatBoost | CCS Replacement | 4 | 0.252302 | 0.677990 | 0.290534 | 0.254971 | 0.271593 | 0.094993 |
| CatBoost | CCS Replacement | 5 | 0.242924 | 0.683246 | 0.289023 | 0.253646 | 0.270181 | 0.095241 |
| CatBoost | Raw + CCS | 1 | 0.251886 | 0.684476 | 0.292044 | 0.256184 | 0.272941 | 0.094827 |
| CatBoost | Raw + CCS | 2 | 0.239276 | 0.674591 | 0.274421 | 0.240937 | 0.256591 | 0.095478 |
| CatBoost | Raw + CCS | 3 | 0.243038 | 0.685362 | 0.277946 | 0.244032 | 0.259887 | 0.095138 |
| CatBoost | Raw + CCS | 4 | 0.253298 | 0.681775 | 0.289023 | 0.253646 | 0.270181 | 0.094860 |
| CatBoost | Raw + CCS | 5 | 0.247085 | 0.687442 | 0.292044 | 0.256297 | 0.273005 | 0.094993 |

## Paired patient bootstrap (2,000 repetitions)

| Model | Treatment | Control | Delta_PR_AUC | PR_AUC_CI95_Lower | PR_AUC_CI95_Upper | Delta_AUROC | AUROC_CI95_Lower | AUROC_CI95_Upper | Delta_F1 | F1_CI95_Lower | F1_CI95_Upper | Delta_Brier | Brier_CI95_Lower | Brier_CI95_Upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | CCS Replacement | Raw | 0.000530 | -0.002125 | 0.003031 | 0.003692 | 0.000836 | 0.006365 | -0.000189 | -0.004566 | 0.005898 | -0.000199 | -0.000377 | -0.000033 |
| LR | Raw + CCS | Raw | -0.001067 | -0.001696 | -0.000450 | -0.001504 | -0.002263 | -0.000835 | -0.001506 | -0.003094 | 0.001094 | 0.000124 | 0.000086 | 0.000166 |
| LR | Raw + CCS | CCS Replacement | -0.001597 | -0.004065 | 0.000959 | -0.005196 | -0.007815 | -0.002511 | -0.001317 | -0.006861 | 0.003536 | 0.000323 | 0.000159 | 0.000499 |
| RF | CCS Replacement | Raw | 0.002419 | -0.000602 | 0.005284 | 0.001689 | -0.000020 | 0.003367 | 0.002165 | -0.002214 | 0.007107 | -0.000697 | -0.000784 | -0.000618 |
| RF | Raw + CCS | Raw | -0.001682 | -0.003638 | 0.000300 | -0.000678 | -0.001852 | 0.000568 | -0.003013 | -0.007291 | 0.001556 | 0.000099 | 0.000071 | 0.000127 |
| RF | Raw + CCS | CCS Replacement | -0.004101 | -0.006909 | -0.001145 | -0.002368 | -0.003913 | -0.000645 | -0.005178 | -0.010003 | -0.000558 | 0.000796 | 0.000712 | 0.000882 |
| XGBoost | CCS Replacement | Raw | 0.001287 | -0.001048 | 0.003608 | 0.001931 | 0.000550 | 0.003364 | 0.001035 | -0.002982 | 0.005322 | -0.000210 | -0.000309 | -0.000107 |
| XGBoost | Raw + CCS | Raw | 0.003726 | 0.002220 | 0.005264 | 0.003545 | 0.002443 | 0.004696 | 0.004706 | 0.000779 | 0.008487 | -0.000342 | -0.000426 | -0.000262 |
| XGBoost | Raw + CCS | CCS Replacement | 0.002440 | 0.000419 | 0.004543 | 0.001613 | 0.000472 | 0.002814 | 0.003672 | -0.000375 | 0.006973 | -0.000132 | -0.000223 | -0.000044 |
| HGB | CCS Replacement | Raw | -0.001430 | -0.004007 | 0.001214 | -0.000717 | -0.002437 | 0.000911 | -0.000471 | -0.005174 | 0.004252 | 0.000078 | -0.000037 | 0.000196 |
| HGB | Raw + CCS | Raw | 0.001932 | 0.000493 | 0.003435 | 0.001400 | 0.000263 | 0.002448 | 0.004613 | 0.000407 | 0.008233 | -0.000103 | -0.000174 | -0.000033 |
| HGB | Raw + CCS | CCS Replacement | 0.003362 | 0.000919 | 0.005824 | 0.002116 | 0.000684 | 0.003616 | 0.005084 | -0.000059 | 0.009165 | -0.000181 | -0.000294 | -0.000071 |
| CatBoost | CCS Replacement | Raw | -0.002277 | -0.004104 | -0.000214 | -0.003721 | -0.005126 | -0.002380 | -0.004049 | -0.007995 | 0.000078 | 0.000150 | 0.000052 | 0.000247 |
| CatBoost | Raw + CCS | Raw | 0.000410 | -0.000782 | 0.001849 | -0.000110 | -0.000825 | 0.000555 | -0.000095 | -0.003741 | 0.002976 | -0.000025 | -0.000084 | 0.000032 |
| CatBoost | Raw + CCS | CCS Replacement | 0.002687 | 0.001014 | 0.004291 | 0.003611 | 0.002498 | 0.004757 | 0.003954 | -0.000098 | 0.007280 | -0.000175 | -0.000257 | -0.000094 |

## Feature counts

| Model | Representation | Fold | Input_Features | Encoded_Features |
| --- | --- | --- | --- | --- |
| CatBoost | CCS Replacement | 1 | 46 | model-native; not fixed |
| CatBoost | CCS Replacement | 2 | 46 | model-native; not fixed |
| CatBoost | CCS Replacement | 3 | 46 | model-native; not fixed |
| CatBoost | CCS Replacement | 4 | 46 | model-native; not fixed |
| CatBoost | CCS Replacement | 5 | 46 | model-native; not fixed |
| CatBoost | Raw | 1 | 46 | model-native; not fixed |
| CatBoost | Raw | 2 | 46 | model-native; not fixed |
| CatBoost | Raw | 3 | 46 | model-native; not fixed |
| CatBoost | Raw | 4 | 46 | model-native; not fixed |
| CatBoost | Raw | 5 | 46 | model-native; not fixed |
| CatBoost | Raw + CCS | 1 | 49 | model-native; not fixed |
| CatBoost | Raw + CCS | 2 | 49 | model-native; not fixed |
| CatBoost | Raw + CCS | 3 | 49 | model-native; not fixed |
| CatBoost | Raw + CCS | 4 | 49 | model-native; not fixed |
| CatBoost | Raw + CCS | 5 | 49 | model-native; not fixed |
| HGB | CCS Replacement | 1 | 46 | 356 |
| HGB | CCS Replacement | 2 | 46 | 355 |
| HGB | CCS Replacement | 3 | 46 | 361 |
| HGB | CCS Replacement | 4 | 46 | 356 |
| HGB | CCS Replacement | 5 | 46 | 357 |
| HGB | Raw | 1 | 46 | 516 |
| HGB | Raw | 2 | 46 | 519 |
| HGB | Raw | 3 | 46 | 519 |
| HGB | Raw | 4 | 46 | 517 |
| HGB | Raw | 5 | 46 | 520 |
| HGB | Raw + CCS | 1 | 49 | 700 |
| HGB | Raw + CCS | 2 | 49 | 703 |
| HGB | Raw + CCS | 3 | 49 | 707 |
| HGB | Raw + CCS | 4 | 49 | 703 |
| HGB | Raw + CCS | 5 | 49 | 705 |
| LR | CCS Replacement | 1 | 46 | 848 |
| LR | CCS Replacement | 2 | 46 | 851 |
| LR | CCS Replacement | 3 | 46 | 850 |
| LR | CCS Replacement | 4 | 46 | 850 |
| LR | CCS Replacement | 5 | 46 | 850 |
| LR | Raw | 1 | 46 | 2416 |
| LR | Raw | 2 | 46 | 2402 |
| LR | Raw | 3 | 46 | 2414 |
| LR | Raw | 4 | 46 | 2418 |
| LR | Raw | 5 | 46 | 2405 |
| LR | Raw + CCS | 1 | 49 | 3022 |
| LR | Raw + CCS | 2 | 49 | 3012 |
| LR | Raw + CCS | 3 | 49 | 3026 |
| LR | Raw + CCS | 4 | 49 | 3030 |
| LR | Raw + CCS | 5 | 49 | 3013 |
| RF | CCS Replacement | 1 | 46 | 848 |
| RF | CCS Replacement | 2 | 46 | 851 |
| RF | CCS Replacement | 3 | 46 | 850 |
| RF | CCS Replacement | 4 | 46 | 850 |
| RF | CCS Replacement | 5 | 46 | 850 |
| RF | Raw | 1 | 46 | 2416 |
| RF | Raw | 2 | 46 | 2402 |
| RF | Raw | 3 | 46 | 2414 |
| RF | Raw | 4 | 46 | 2418 |
| RF | Raw | 5 | 46 | 2405 |
| RF | Raw + CCS | 1 | 49 | 3022 |
| RF | Raw + CCS | 2 | 49 | 3012 |
| RF | Raw + CCS | 3 | 49 | 3026 |
| RF | Raw + CCS | 4 | 49 | 3030 |
| RF | Raw + CCS | 5 | 49 | 3013 |
| XGBoost | CCS Replacement | 1 | 46 | 848 |
| XGBoost | CCS Replacement | 2 | 46 | 851 |
| XGBoost | CCS Replacement | 3 | 46 | 850 |
| XGBoost | CCS Replacement | 4 | 46 | 850 |
| XGBoost | CCS Replacement | 5 | 46 | 850 |
| XGBoost | Raw | 1 | 46 | 2416 |
| XGBoost | Raw | 2 | 46 | 2402 |
| XGBoost | Raw | 3 | 46 | 2414 |
| XGBoost | Raw | 4 | 46 | 2418 |
| XGBoost | Raw | 5 | 46 | 2405 |
| XGBoost | Raw + CCS | 1 | 49 | 3022 |
| XGBoost | Raw + CCS | 2 | 49 | 3012 |
| XGBoost | Raw + CCS | 3 | 49 | 3026 |
| XGBoost | Raw + CCS | 4 | 49 | 3030 |
| XGBoost | Raw + CCS | 5 | 49 | 3013 |

CatBoost uses model-native categorical handling; its encoded dimension is not a fixed one-hot count.

## Controlled training settings

- LR/RF/XGBoost: `run_xgb_5fold_recheck.py`; train-fold numeric median imputation and scaling, categorical mode imputation and one-hot encoding. LR: C=1, lbfgs, 1500 iterations. RF: 300 trees, max depth 16, minimum leaf 2. XGBoost: depth 4, learning rate 0.05, 500-tree limit, 30-round early stopping on a patient-disjoint 20% inner validation. All three representations were rerun on CPU with eight threads and no class weighting.
- HGB: `run_diagnosis_representation_hgb.py`; train-fold median/constant imputation, one-hot encoding with minimum frequency 100. Five candidates from `run_nested_hgb_blend.py` are selected by inner-validation AP, then refit on the full outer train.
- CatBoost: saved OOF from `run_ccs_catboost_5fold.py` and `run_paper_fair_comparison.py`; raw diagnosis codes remain categorical. Depths 4/6 and 250/500/750 trees are chosen by inner-validation AP, then refit on the full outer train. CatBoost random seed is 2026 for all three variants.
- CCS mapping: `load_ccs` and `attach_uci_ccs` from the existing project; the three representations have 46, 46 and 49 input columns respectively.
- Every model uses the saved seed-777 patient-disjoint outer folds. Precision/Recall/F1 use the highest 10% test risks in each fold. No external data, class weight, SMOTE or SMOTENC are used.
