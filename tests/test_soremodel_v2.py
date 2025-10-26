"""
Teste completo e organizado do SOREModel v2
Demonstra o uso completo dos módulos: tokenizer, modelo, treinamento e geração
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from core import criar_tokenizer_basico
from models import SOREModel_v2
from training import Trainer
from generation import TextGenerator


def test_tokenizer():
    """Testa as funcionalidades do tokenizer"""
    print("=== Testando Tokenizer ===")

    # Criar tokenizer
    tokenizer = criar_tokenizer_basico()

    # Testar codificação/decodificação
    texto_teste = "olá, mundo!"
    codificado = tokenizer.codificar(texto_teste)
    decodificado = tokenizer.decodificar(codificado)

    print(f"Texto original: {texto_teste}")
    print(f"Codificado: {codificado}")
    print(f"Decodificado: {decodificado}")
    print(f"Tamanho do vocabulário: {tokenizer.get_vocab_size()}")

    # Testar preparação de dados
    textos_treinamento = ["olá, mundo!", "oi, tchau!", "abc", "def"]
    X_batch, Y_batch = tokenizer.encode_batch(textos_treinamento, contexto_tamanho=3)
    print(f"Exemplos de treinamento: X={len(X_batch)}, Y={len(Y_batch)}")

    return tokenizer


def test_modelo():
    """Testa a criação e funcionalidades básicas do modelo"""
    print("\n=== Testando Modelo ===")

    # Parâmetros do modelo
    tamanho_vocab = 100  # Vocabulário básico
    dim_embed = 32
    tamanho_contexto = 8
    num_heads = 4
    num_layers = 2

    # Criar modelo
    modelo = SOREModel_v2(
        tamanho_vocab=tamanho_vocab,
        dim_embed=dim_embed,
        tamanho_contexto=tamanho_contexto,
        num_heads=num_heads,
        num_layers=num_layers
    )

    print(f"Modelo criado com {sum(p.numel() for p in modelo.parameters())} parâmetros")

    # Testar forward pass com dados dummy
    batch_size = 4
    x_teste = torch.randint(0, tamanho_vocab, (batch_size, tamanho_contexto))
    with torch.no_grad():
        saida = modelo(x_teste)
        print(f"Shape da saída: {saida.shape}")

    return modelo


def test_treinamento():
    """Testa o treinamento do modelo"""
    print("\n=== Testando Treinamento ===")

    # Preparar dados
    tokenizer = criar_tokenizer_basico()
    textos_treinamento = [
        "olá, mundo!",
        "oi, tchau!",
        "como você está?",
        "estou bem!",
        "que bom!",
        "sim, muito bom!",
        "adeus!",
        "até logo!"
    ]

    # Criar modelo
    modelo = SOREModel_v2(
        tamanho_vocab=tokenizer.get_vocab_size(),
        dim_embed=32,
        tamanho_contexto=8,
        num_heads=4,
        num_layers=2
    )

    # Criar trainer
    trainer = Trainer(modelo, tokenizer)

    # Treinar
    contexto_tamanho = 8
    trainer.treinar(
        textos=textos_treinamento,
        contexto_tamanho=contexto_tamanho,
        num_epocas=200,
        batch_size=4,
        learning_rate=0.001
    )

    print(f"Perda final: {trainer.get_historico_perdas()[-1]:.4f}")

    return modelo, trainer


def test_geracao():
    """Testa a geração de texto"""
    print("\n=== Testando Geração de Texto ===")

    # Preparar modelo e tokenizer
    tokenizer = criar_tokenizer_basico()
    modelo = SOREModel_v2(
        tamanho_vocab=tokenizer.get_vocab_size(),
        dim_embed=32,
        tamanho_contexto=8,
        num_heads=4,
        num_layers=2
    )

    # Criar trainer e treinar rapidamente
    trainer = Trainer(modelo, tokenizer)
    textos_treinamento = ["olá", "oi", "tchau", "adeus", "sim", "não"]
    trainer.treinar(textos_treinamento, contexto_tamanho=3, num_epocas=100, batch_size=2)

    # Criar gerador
    gerador = TextGenerator(modelo, tokenizer)

    # Testar geração
    contexto_inicial = "olá"
    texto_gerado = gerador.gerar_texto(contexto_inicial, max_length=10, temperature=0.8)
    print(f"Contexto inicial: '{contexto_inicial}'")
    print(f"Texto gerado: '{texto_gerado}'")

    # Testar variações
    variacoes = gerador.gerar_variacoes("oi", num_variacoes=3, max_length=5, temperature=1.2)
    print("Variações geradas:")
    for i, var in enumerate(variacoes, 1):
        print(f"  {i}: '{var}'")

    return gerador


def test_completar_texto():
    """Testa a funcionalidade de completar texto"""
    print("\n=== Testando Completação de Texto ===")

    # Preparar modelo
    tokenizer = criar_tokenizer_basico()
    modelo = SOREModel_v2(
        tamanho_vocab=tokenizer.get_vocab_size(),
        dim_embed=32,
        tamanho_contexto=8,
        num_heads=4,
        num_layers=2
    )

    # Treinar com textos simples
    trainer = Trainer(modelo, tokenizer)
    textos = ["abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"]
    trainer.treinar(textos, contexto_tamanho=2, num_epocas=100, batch_size=2)

    # Testar completação
    gerador = TextGenerator(modelo, tokenizer)
    texto_incompleto = "ab"
    texto_completo = gerador.completar_texto(texto_incompleto, max_completar=5)
    print(f"Texto incompleto: '{texto_incompleto}'")
    print(f"Texto completo: '{texto_completo}'")


def test_salvar_carregar():
    """Testa salvar e carregar modelo"""
    print("\n=== Testando Salvar/Carregar Modelo ===")

    # Preparar e treinar modelo
    tokenizer = criar_tokenizer_basico()
    modelo = SOREModel_v2(
        tamanho_vocab=tokenizer.get_vocab_size(),
        dim_embed=32,
        tamanho_contexto=8,
        num_heads=4,
        num_layers=2
    )

    trainer = Trainer(modelo, tokenizer)
    textos = ["hello", "world", "test", "save", "load"]
    trainer.treinar(textos, contexto_tamanho=3, num_epocas=50, batch_size=2)

    # Salvar modelo
    trainer.salvar_modelo("modelo_teste.pth")

    # Criar novo modelo e carregar
    modelo_novo = SOREModel_v2(
        tamanho_vocab=tokenizer.get_vocab_size(),
        dim_embed=32,
        tamanho_contexto=8,
        num_heads=4,
        num_layers=2
    )

    trainer_novo = Trainer(modelo_novo, tokenizer)
    trainer_novo.carregar_modelo("modelo_teste.pth")

    print("Modelo salvo e carregado com sucesso!")

    # Testar se funciona
    gerador = TextGenerator(modelo_novo, tokenizer)
    texto = gerador.gerar_texto("he", max_length=5)
    print(f"Geração após carregamento: '{texto}'")


def run_all_tests():
    """Executa todos os testes"""
    print("🚀 Iniciando testes completos do SOREModel v2\n")

    try:
        # Executar todos os testes
        tokenizer = test_tokenizer()
        modelo = test_modelo()
        modelo_treinado, trainer = test_treinamento()
        gerador = test_geracao()
        test_completar_texto()
        test_salvar_carregar()

        print("\n✅ Todos os testes executados com sucesso!")
        print("\n📊 Resumo dos testes:")
        print(f"   - Tokenizer: {tokenizer.get_vocab_size()} caracteres")
        print(f"   - Modelo: {sum(p.numel() for p in modelo_treinado.parameters())} parâmetros")
        print(f"   - Treinamento: {len(trainer.get_historico_perdas())} épocas")
        print(f"   - Geração: {len(gerador.gerar_texto('teste', max_length=5))} caracteres gerados")

    except Exception as e:
        print(f"\n❌ Erro durante os testes: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
