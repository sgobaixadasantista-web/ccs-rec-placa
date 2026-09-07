# CCS Reconhecimento de Placas

Protótipo para reconhecimento de placas veiculares brasileiras usando **YOLOv11 + OpenCV + EasyOCR**.

## Fluxo atual

1. Receber uma imagem do veículo.
2. YOLO localizar automaticamente a placa.
3. Recortar a região detectada.
4. Melhorar contraste e gerar múltiplos pré-processamentos.
5. Executar OCR com EasyOCR.
6. Normalizar e corrigir caracteres ambíguos conforme a posição.
7. Validar os formatos brasileiros:
   - Antigo: `ABC1234`
   - Mercosul: `ABC1D23`
8. Retornar placa, modelo, bbox e confianças.

A integração com Supabase será adicionada depois que o reconhecimento estiver confiável.

## Modelo YOLO padrão

O projeto usa por padrão o modelo público:

`felipedutrain/placa-br-yolov11`

Hugging Face:

`https://huggingface.co/felipedutrain/placa-br-yolov11`

O modelo localiza placas brasileiras. A leitura dos caracteres é feita pelo EasyOCR.

Arquivo usado automaticamente:

```text
https://huggingface.co/felipedutrain/placa-br-yolov11/resolve/main/best.pt
```

## OCR

O pipeline principal passou a usar **EasyOCR** em vez do Tesseract.

São testadas várias versões do mesmo recorte:

- imagem ampliada;
- escala de cinza;
- CLAHE;
- Otsu;
- limiarização adaptativa.

As leituras são classificadas pelos padrões brasileiros e o melhor candidato é retornado.

O `reconhecer_placa.py` antigo continua disponível como teste legado com Tesseract, por isso `pytesseract` permanece nas dependências por enquanto.

Na primeira execução, o EasyOCR pode baixar os pesos do modelo de reconhecimento automaticamente.

## Instalação

```bash
pip install -r requirements.txt
```

## Executar reconhecimento completo

```bash
python detector_yolo.py foto.jpg --salvar-recortes
```

Também é possível usar um modelo YOLO local:

```bash
python detector_yolo.py foto.jpg --modelo weights/license_plate.pt --salvar-recortes
```

Exemplo de retorno esperado:

```json
[
  {
    "bbox": [310, 240, 505, 292],
    "confianca_yolo": 0.94,
    "modelo_detector": "https://huggingface.co/felipedutrain/placa-br-yolov11/resolve/main/best.pt",
    "placa": "UEQ0F29",
    "modelo": "MERCOSUL",
    "valida": true,
    "ocr_bruto": "UEQOF29",
    "confianca_ocr": 0.91,
    "motor_ocr": "easyocr",
    "recorte": "recortes/placa_0_UEQ0F29.jpg"
  }
]
```

## Regras de validação

### Placa antiga

```text
AAA0000
ABC1234
```

### Mercosul

```text
AAA0A00
ABC1D23
```

O sistema também tenta corrigir confusões comuns de OCR de acordo com a posição esperada, como:

- `O` / `0`
- `I` / `1`
- `Q` / `0`
- `G` / `6`
- `B` / `8`
- `S` / `5`
- `Z` / `2`

## Fine-tuning próprio

Mesmo usando o modelo pré-treinado, podemos melhorar o desempenho com imagens próprias do ambiente operacional.

A estrutura esperada é:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

A classe YOLO continua sendo apenas:

```text
license_plate
```

A distinção entre placa antiga e Mercosul é feita no OCR/validador.

## Status

Já temos:

- detector YOLOv11 pré-treinado para placas brasileiras;
- EasyOCR como OCR principal;
- múltiplos pré-processamentos por placa;
- correção por máscara para placa antiga e Mercosul;
- retorno de `bbox`, confiança YOLO, confiança OCR e motor OCR;
- opção para salvar o recorte detectado;
- estrutura pronta para fine-tuning posterior;
- caminho aberto para integração com Supabase.

## Próximas etapas

- testar YOLO + EasyOCR em imagens reais;
- medir precisão do detector e OCR separadamente;
- adicionar votação entre múltiplas leituras quando necessário;
- processar vídeo/câmera em tempo real;
- salvar eventos válidos no Supabase.
