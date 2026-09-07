# Guia de anotacao do dataset

O detector YOLO deste projeto usa uma unica classe:

```text
0 = license_plate
```

## O que deve ser marcado

Marque somente o retangulo da placa do veiculo, sem incluir para-choque, grade, farol ou grandes margens externas.

Valem os dois padroes brasileiros:

- placa antiga: `ABC1234`
- placa Mercosul: `ABC1D23`

O YOLO nao precisa saber qual dos dois modelos e; ele aprende apenas a localizar a placa. A classificacao do modelo e feita depois pelo OCR.

## Formato YOLO

Para cada imagem `foto001.jpg`, crie um arquivo `foto001.txt` com:

```text
0 x_center y_center width height
```

Todos os valores de coordenadas devem estar normalizados entre 0 e 1.

Exemplo:

```text
0 0.503125 0.612500 0.225000 0.095000
```

## Recomendacoes para o primeiro dataset

Para um MVP funcional, tente reunir pelo menos 100 a 300 imagens variadas. Para melhorar robustez, aumente gradualmente para 1.000 ou mais.

Inclua variacao de:

- placa antiga e Mercosul;
- dia e noite;
- chuva e tempo seco;
- placas proximas e distantes;
- frente e traseira;
- diferentes angulos;
- carros, motos, vans e caminhoes;
- iluminacao forte, sombra e reflexo;
- fotos de celular e cameras fixas.

Evite colocar placas ou rostos de terceiros em repositorios publicos sem avaliar privacidade e finalidade de uso. Para dados operacionais reais, prefira manter o dataset privado.

## Estrutura esperada

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Depois de preparar o dataset, execute:

```bash
python treinar_yolo.py
```

O melhor peso sera salvo pelo Ultralytics em algo como:

```text
runs/placas/weights/best.pt
```
