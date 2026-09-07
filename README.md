# CCS Reconhecimento de Placas

Protótipo para reconhecimento de placas veiculares brasileiras usando YOLO + OpenCV + Tesseract OCR.

## Objetivo

Fluxo atual:

1. Receber uma imagem do veículo.
2. YOLO localizar automaticamente a placa.
3. Recortar a região detectada.
4. Melhorar contraste e gerar múltiplos pré-processamentos.
5. Executar OCR.
6. Normalizar e corrigir caracteres ambíguos conforme a posição.
7. Validar os formatos brasileiros:
   - Antigo: `ABC1234`
   - Mercosul: `ABC1D23`
8. Retornar placa, modelo, bbox e confianças.

A integração com Supabase será adicionada depois que o reconhecimento estiver confiável.

## Instalação

```bash
pip install -r requirements.txt
```

Também é necessário instalar o Tesseract OCR no sistema operacional.

### Windows

Se necessário, configure:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

## YOLO

O arquivo `detector_yolo.py` espera um modelo treinado especificamente para detectar placas.

Caminho padrão:

```text
weights/license_plate.pt
```

Esse peso ainda precisa ser treinado ou adicionado ao projeto. Um modelo YOLO genérico não é suficiente para localizar placas com boa confiabilidade.

### Estrutura esperada do dataset

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Cada imagem deve ter um arquivo `.txt` correspondente no formato YOLO:

```text
0 x_center y_center width height
```

A classe `0` representa `license_plate`.

## Treinar o detector

```bash
python treinar_yolo.py
```

O script inicia o fine-tuning a partir de `yolo11n.pt` e usa `dataset.yaml`.

Ao final, copie o melhor peso para:

```text
weights/license_plate.pt
```

Exemplo:

```bash
mkdir weights
cp runs/placas/weights/best.pt weights/license_plate.pt
```

No Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force weights
Copy-Item runs/placas/weights/best.pt weights/license_plate.pt
```

## Executar reconhecimento completo

```bash
python detector_yolo.py foto.jpg --modelo weights/license_plate.pt --salvar-recortes
```

Exemplo de retorno:

```json
[
  {
    "bbox": [310, 240, 505, 292],
    "confianca_yolo": 0.94,
    "placa": "UEQ0F29",
    "modelo": "MERCOSUL",
    "valida": true,
    "ocr_bruto": "UEQOF29",
    "confianca_ocr": 0.91,
    "recorte": "recortes/placa_0_UEQ0F29.jpg"
  }
]
```

## Reconhecimento sem YOLO

O arquivo `reconhecer_placa.py` permanece disponível para testes de OCR com uma região conhecida.

```bash
python reconhecer_placa.py foto.jpg
```

## Status

Já validamos em uma imagem real que o OCR consegue chegar próximo da placa correta, mas há confusões entre caracteres semelhantes como:

- `O` / `0`
- `I` / `1`
- `Q` / `0`
- `G` / `6`
- `B` / `8`

O novo pipeline aplica correções condicionadas ao padrão esperado de cada modelo de placa.

### Próximas etapas

- montar dataset de placas reais;
- treinar o YOLO detector de placas;
- testar em fotos de diferentes distâncias, ângulos e iluminação;
- medir precisão do detector e do OCR separadamente;
- integrar eventos válidos ao Supabase;
- posteriormente processar vídeo/câmera em tempo real.
