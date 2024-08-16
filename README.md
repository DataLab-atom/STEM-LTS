```
git clone git@github.com:DC-research/TEMPO.git
mv TEMPO/* ./
python domain.py
CUDA_VISIBLE_DEVICES=5,6,7 python domain.py  --datasets electricity --pred_len 192 > TEMPO_6_96_electricity.log 2>&1 
```