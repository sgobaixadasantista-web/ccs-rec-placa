# CCS Reconhecimento de Placas

Protótipo para reconhecimento de placas veiculares brasileiras usando **YOLOv11 + OpenCV + Tesseract OCR**.

## Fluxo atual

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

## Modelo YOLO padrão

O projeto passa a usar por padrão o modelo público:

`felipedutrain/placa-br-yolov11`

Hugging Face:

`https://huggingface.co/felipedutrain/placa-br-yolov11`

O modelo foi treinado para detectar placas brasileiras Mercosul e antigas. Ele localiza a placa; a leitura dos caracteres continua sendo feita pelo nosso OCR.

Arquivo usado automaticamente:

```text
https://huggingface.co/felipedutrain/placa-br-yolov11/resolve/main/best.pt
```

Licença informada pelo repositório do modelo: MIT.

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

## Executar reconhecimento completo

Agora não é necessário informar manualmente um `best.pt` para o primeiro teste:

```bash
python detector_yolo.py foto.jpg --salvar-recortes
```

O Ultralytics carrega o peso público definido em `MODELO_PADRAO`.

Também é possível usar um modelo local ou outro peso:

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
    "recorte": "recortes/placa_0_UEQ0F29.jpg"
  }
]
```

## Reconhecimento sem YOLO

O arquivo `reconhecer_placa.py` permanece disponível para testes de OCR com uma região conhecida.

```bash
python reconhecer_placa.py foto.jpg
```

## Fine-tuning próprio

Mesmo usando o modelo pré-treinado, continuamos podendo melhorar o desempenho com imagens próprias do ambiente operacional.

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

- detector de placas brasileiras pré-treinado;
- OCR com Tesseract;
- correção por máscara para placa antiga e Mercosul;
- retorno de `bbox`, confiança YOLO e confiança OCR;
- opção para salvar o recorte detectado;
- estrutura pronta para fine-tuning posterior;
- caminho aberto para integração com Supabase.

## Próximas etapas

- testar o modelo pré-treinado em nossas imagens reais;
- medir precisão do detector e OCR separadamente;
- melhorar OCR para caracteres ambíguos;
- processar vídeo/câmera em tempo real;
- salvar eventos válidos no Supabase.
