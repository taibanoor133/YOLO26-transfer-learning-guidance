# YOLO26 Made Simple

A beginner-friendly guide (easy English) to **YOLO26**, **Transfer Learning**, **Fine-Tuning** and **Incremental Learning**, with practical Ultralytics code.

## What is inside
- `GUIDE.md`: the full guide (read it right here on GitHub)
- `docs/YOLO26_Complete_Guide.pdf`: the full guide as a 35-page PDF
- `docs/YOLO26_Made_Simple_Carousel.pdf`: a 12-slide visual summary

## What you will learn
1. How YOLO works (Backbone, Neck, Head)
2. What is new in YOLO26 (no NMS, no DFL, built for the edge)
3. Transfer Learning: train YOLO on your own classes
4. Fine-Tuning: two-stage training with a small learning rate
5. Incremental Learning: add new classes without forgetting old ones
6. Evaluation, export, troubleshooting and an 8-week roadmap

## Quick start
```bash
pip install -U ultralytics
```
```python
from ultralytics import YOLO
model = YOLO("yolo26s.pt")
model.train(data="data.yaml", epochs=50, imgsz=640)
```

## Notes
- Code was written from the official Ultralytics docs. Test it on your own data before relying on it.
- Check https://docs.ultralytics.com/models/yolo26/ for the latest YOLO26 details.

## Author
Your Name | [LinkedIn](https://www.linkedin.com/in/your-profile)

If this helped you, give the repo a star ⭐
