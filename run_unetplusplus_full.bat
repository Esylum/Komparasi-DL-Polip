@echo off
docker compose run --rm polyp python "unet++.py" --loss all --epochs 30 --image-size 224 --batch-size 8 --base-filters 16
pause
