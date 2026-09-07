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

O pipeline principal usa **EasyOCR**.

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

## Benchmark com UFPR-ALPR

O projeto agora possui o script:

```text
converter_ufpr_yolo.py
```

Ele converte as anotações locais do UFPR-ALPR para o formato YOLO usando uma única classe:

```text
license_plate
```

Exemplo:

```bash
python converter_ufpr_yolo.py \
  --origem /caminho/UFPR-ALPR \
  --destino dataset/ufpr_yolo
```

O conversor tenta preservar os conjuntos `train`, `validation` e `test` quando presentes e gera:

```text
dataset/ufpr_yolo/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  dataset_ufpr.yaml
```

**Importante:** as imagens do UFPR-ALPR não devem ser adicionadas ou redistribuídas neste repositório. O dataset possui termos próprios de uso para pesquisa acadêmica/não comercial. Este projeto mantém apenas o conversor. Veja `docs/UFPR_ALPR.md`.

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
- conversor UFPR-ALPR -> YOLO para benchmark/fine-tuning local;
- estrutura pronta para fine-tuning posterior;
- caminho aberto para integração com Supabase.

## Próximas etapas

- obter acesso autorizado ao UFPR-ALPR ou outro dataset compatível;
- converter e validar visualmente as anotações;
- testar YOLO + EasyOCR em lote;
- medir precisão do detector e OCR separadamente;
- adicionar votação entre múltiplas leituras quando necessário;
- processar vídeo/câmera em tempo real;
- salvar eventos válidos no Supabase.
