"""
Exemplo simples de uso do SOREModel v2
Demonstra o uso básico: tokenizer, modelo, treinamento e geração
"""

import sys
import os

# Adicionar src ao path para importar os módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src import SOREModel_v2, criar_tokenizer_basico, Trainer, TextGenerator


def main():
    """Função principal do exemplo"""
    print("🚀 Exemplo de uso do SOREModel v2")
    print("=" * 40)

    # 1. Preparar dados de treinamento
    print("\n📚 Preparando dados de treinamento...")
    tokenizer = criar_tokenizer_basico()

    textos_treinamento = [
        "olá, mundo!",
        "como você está?",
        "estou bem, obrigado!",
        "que bom ouvir isso!",
        "sim, muito bom!",
        "adeus, até logo!",
        "oi, tudo bem?",
        "tudo ótimo!",
        "nos vemos em breve",
        "até a próxima!"
    ]

    print(f"✅ Tokenizer criado com vocabulário de {tokenizer.get_vocab_size()} caracteres")
    print(f"✅ {len(textos_treinamento)} textos de treinamento preparados")

    # 2. Criar modelo
    print("\n🧠 Criando modelo...")
    modelo = SOREModel_v2(
        tamanho_vocab=tokenizer.get_vocab_size(),
        dim_embed=64,                                                                           # Dimensão maior para melhor qualidade
        tamanho_contexto=16,                                                                    # Contexto maior
        num_heads=8,                                                                            # Mais cabeças de atenção
        num_layers=4                                                                            # Mais camadas
    )

    print("✅ Modelo criado com sucesso!")
    print(f"   - Vocabulário: {tokenizer.get_vocab_size()} tokens")
    print(f"   - Embedding: {64} dimensões")
    print(f"   - Contexto: {16} tokens")
    print(f"   - Atenção: {8} cabeças")
    print(f"   - Camadas: {4} blocos Transformer")

    # 3. Treinar modelo
    print("\n🎯 Iniciando treinamento...")
    trainer = Trainer(modelo, tokenizer)

    trainer.treinar(
        textos=textos_treinamento,
        contexto_tamanho=8,
        num_epocas=500,
        batch_size=4,
        learning_rate=0.001
    )

    print(f"✅ Treinamento concluído! Perda final: {trainer.get_historico_perdas()[-1]:.4f}")
    # 4. Gerar texto
    print("\n✍️  Gerando texto...")
    gerador = TextGenerator(modelo, tokenizer)

    # Testar diferentes contextos iniciais
    contextos_teste = ["olá", "oi", "como", "sim", "adeus"]

    for contexto in contextos_teste:
        texto_gerado = gerador.gerar_texto(
            contexto_inicial=contexto,
            max_length=15,
            temperature=0.8,                                                                    # Equilíbrio entre criatividade e coerência
            top_p=0.9                                                                           # Nucleus sampling para qualidade
        )
        print(f"  '{contexto}' → '{texto_gerado}'")

    # 5. Demonstrar ajuste fino
    print("\n🔧 Testando ajuste fino...")
    novos_textos = [
        "python é incrível!",
        "programação é divertido",
        "código limpo é importante"
    ]

    trainer.ajustar_fino(
        textos_novos=novos_textos,
        contexto_tamanho=8,
        num_epocas=200
    )

    # Gerar com o modelo ajustado
    texto_programacao = gerador.gerar_texto("python", max_length=10, temperature=0.7)
    print(f"  Após ajuste fino: '{texto_programacao}'")

    # 6. Salvar modelo
    print("\n💾 Salvando modelo...")
    trainer.salvar_modelo("exemplo_soremodel.pth")
    print("✅ Modelo salvo com sucesso!")

    print("\n🎉 Exemplo concluído com sucesso!")
    print("\n📊 Resumo:")
    print(f"   - Textos de treinamento: {len(textos_treinamento)}")
    print(f"   - Novos textos para ajuste: {len(novos_textos)}")
    print(f"   - Épocas de treinamento: 500")
    print(f"   - Épocas de ajuste fino: 200")
    print(f"   - Perda final: {trainer.get_historico_perdas()[-1]:.4f}")
    print("\n💡 Dicas:")
    print("   - Aumentar num_epocas para melhor qualidade")
    print("   - Experimentar diferentes temperaturas (0.5-1.2)")
    print("   - Usar mais dados para resultados melhores")


if __name__ == "__main__":
    main()
