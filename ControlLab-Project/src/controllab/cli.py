

import sys
import argparse
import sympy as sp

# Importação explícita dos submódulos principais
import controllab.analysis
import controllab.core
import controllab.design
import controllab.modeling
import controllab.numerical

# Imports explícitos para CLI manual
from controllab.core.symbolic_tf import SymbolicTransferFunction
from controllab.analysis.stability_analysis import (
    analyze_stability,
    quick_stability_check,
    compare_systems_stability,
    run_module_validation,
    StabilityAnalysisEngine
)
from controllab.analysis.frequency_response import calculate_frequency_response
from controllab.analysis.root_locus import get_locus_features
from controllab.analysis.routh_hurwitz import build_routh_array
from controllab.analysis.stability_utils import (
    validate_stability_methods,
    cross_validate_poles,
    format_stability_report,
    stability_region_2d,
    root_locus_3d,
    analyze_sensitivity
)


def main():
    parser = argparse.ArgumentParser(description="ControlLab CLI - Ferramentas de análise e projeto de sistemas de controle")
    subparsers = parser.add_subparsers(dest="command", help="Comando a executar")

    # Comando: estabilidade
    parser_stab = subparsers.add_parser("estabilidade", help="Análise de estabilidade de uma função de transferência")
    parser_stab.add_argument("ft", type=str, help="Função de transferência, exemplo: '1/(s**2+2*s+1)'")
    parser_stab.add_argument("--relatorio", action="store_true", help="Exibe relatório pedagógico completo")

    # Comando: bode
    parser_bode = subparsers.add_parser("bode", help="Plotar diagrama de Bode")
    parser_bode.add_argument("ft", type=str, help="Função de transferência, exemplo: '1/(s+1)'")

    # Comando: root_locus
    parser_rl = subparsers.add_parser("root_locus", help="Plotar lugar das raízes")
    parser_rl.add_argument("ft", type=str, help="Função de transferência, exemplo: '1/(s+1)'")

    # Comando: routh
    parser_routh = subparsers.add_parser("routh", help="Tabela de Routh-Hurwitz")
    parser_routh.add_argument("poly", type=str, help="Polinômio característico, exemplo: 's**3 + 2*s**2 + s + 1'")
    parser_routh.add_argument("--pedagogico", action="store_true", help="Exibe relatório pedagógico completo da análise de Routh")

    # Comando: quick_stability
    parser_quick = subparsers.add_parser("quick_stability", help="Verificação rápida de estabilidade")
    parser_quick.add_argument("ft", type=str, help="Função de transferência")

    # Comando: comparar
    parser_comp = subparsers.add_parser("comparar", help="Comparação de estabilidade entre múltiplos sistemas")
    parser_comp.add_argument("fts", nargs='+', help="Lista de funções de transferência para comparar")

    # Comando: validar_modulo
    parser_valmod = subparsers.add_parser("validar_modulo", help="Validação completa do módulo de estabilidade")

    # Comando: step_response
    parser_step = subparsers.add_parser("step_response", help="Análise da resposta ao degrau")
    parser_step.add_argument("ft", type=str, help="Função de transferência")

    # Comando: impulse_response
    parser_impulse = subparsers.add_parser("impulse_response", help="Análise da resposta ao impulso")
    parser_impulse.add_argument("ft", type=str, help="Função de transferência")

    # Comando: transient_response
    parser_transient = subparsers.add_parser("transient_response", help="Análise da resposta transitória")
    parser_transient.add_argument("ft", type=str, help="Função de transferência")

    # Comando: compare_responses
    parser_compresp = subparsers.add_parser("compare_responses", help="Comparação de respostas de sistemas")
    parser_compresp.add_argument("fts", nargs='+', help="Lista de funções de transferência para comparar")

    # Comando: validate_methods
    parser_valmethods = subparsers.add_parser("validate_methods", help="Validação cruzada dos métodos de estabilidade")
    parser_valmethods.add_argument("ft", type=str, help="Função de transferência")

    # Comando: cross_validate_poles
    parser_crossval = subparsers.add_parser("cross_validate_poles", help="Validação cruzada dos polos do sistema")
    parser_crossval.add_argument("ft", type=str, help="Função de transferência")

    # Comando: format_stability_report
    parser_formatrep = subparsers.add_parser("format_stability_report", help="Formata relatório de estabilidade")
    parser_formatrep.add_argument("ft", type=str, help="Função de transferência")

    # Comando: stability_region_2d
    parser_region2d = subparsers.add_parser("stability_region_2d", help="Análise paramétrica de estabilidade 2D")
    parser_region2d.add_argument("ft", type=str, help="Função de transferência")
    parser_region2d.add_argument("param1", type=str, help="Parâmetro 1 (ex: 'a')")
    parser_region2d.add_argument("param2", type=str, help="Parâmetro 2 (ex: 'b')")

    # Comando: root_locus_3d
    parser_rl3d = subparsers.add_parser("root_locus_3d", help="Lugar das raízes 3D paramétrico")
    parser_rl3d.add_argument("ft", type=str, help="Função de transferência")
    parser_rl3d.add_argument("param1", type=str, help="Parâmetro 1 (ex: 'a')")
    parser_rl3d.add_argument("param2", type=str, help="Parâmetro 2 (ex: 'b')")

    # Comando: analyze_sensitivity
    parser_sens = subparsers.add_parser("analyze_sensitivity", help="Análise de sensibilidade paramétrica")
    parser_sens.add_argument("ft", type=str, help="Função de transferência")
    parser_sens.add_argument("param", type=str, help="Parâmetro para análise de sensibilidade")

    # Comandos automáticos para todas as funções públicas dos submódulos principais
    import inspect
    import types

    # Importação explícita dos submódulos
    import controllab.analysis
    import controllab.core
    import controllab.design
    import controllab.modeling
    import controllab.numerical

    controllab_modules = [
        controllab.analysis,
        controllab.core,
        controllab.design,
        controllab.modeling,
        controllab.numerical
    ]

    # Função para adicionar comandos para cada função pública
    for module in controllab_modules:
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                parser_func = subparsers.add_parser(name, help=f'Executa {name} do módulo {module.__name__}')
                parser_func.add_argument('--args', nargs='*', help='Argumentos para a função')

    args = parser.parse_args()

    if args.command == "estabilidade":
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        if args.relatorio:
            engine = StabilityAnalysisEngine()
            report = engine.comprehensive_analysis(tf)
            print(report.get_full_report())
        else:
            result = analyze_stability(tf)
            print("Análise de estabilidade:")
            print(result)

    elif args.command == "bode":
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        print("Plotando diagrama de Bode...")
        # Usar função correta do módulo frequency_response
        calculate_frequency_response(tf, omega_range=None)  # omega_range pode ser ajustado conforme necessário

    elif args.command == "root_locus":
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        print("Plotando lugar das raízes...")
        # Usar função correta do módulo root_locus
        get_locus_features(tf)

    elif args.command == "routh":
        s = sp.Symbol('s')
        poly = sp.sympify(args.poly)
        if args.pedagogico:
            from controllab.analysis.routh_hurwitz import RouthHurwitzAnalyzer
            analyzer = RouthHurwitzAnalyzer()
            routh_array = analyzer.build_routh_array(poly, s, show_steps=True)
            result = analyzer.analyze_stability(routh_array, show_steps=True)
            print(result.get_formatted_history())
        else:
            # Usar função correta do módulo routh_hurwitz
            routh_obj = build_routh_array(poly, s)
            print(routh_obj.display_array())

    elif args.command == "quick_stability":
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        is_stable = quick_stability_check(tf)
        print(f"Estável? {'Sim' if is_stable else 'Não'}")

    elif args.command == "comparar":
        s = sp.Symbol('s')
        tfs = [SymbolicTransferFunction(sp.numer(sp.sympify(ft)), sp.denom(sp.sympify(ft))) for ft in args.fts]
        labels = [f"Sistema {i+1}" for i in range(len(tfs))]
        comparison = compare_systems_stability(tfs, labels)
        print("Comparação de estabilidade entre sistemas:")
        print(comparison)

    elif args.command == "validar_modulo":
        print("Executando validação completa do módulo de estabilidade...")
        results = run_module_validation()
        print(results)

    elif args.command == "step_response":
        from controllab.analysis import analyze_step_response
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        result = analyze_step_response(tf)
        print("Resposta ao degrau:")
        print(result)

    elif args.command == "impulse_response":
        from controllab.analysis import analyze_impulse_response
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        result = analyze_impulse_response(tf)
        print("Resposta ao impulso:")
        print(result)

    elif args.command == "transient_response":
        from controllab.analysis import analyze_transient_response
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        result = analyze_transient_response(tf)
        print("Resposta transitória:")
        print(result)

    elif args.command == "compare_responses":
        from controllab.analysis import compare_responses
        s = sp.Symbol('s')
        tfs = [SymbolicTransferFunction(sp.numer(sp.sympify(ft)), sp.denom(sp.sympify(ft))) for ft in args.fts]
        result = compare_responses(*tfs)
        print("Comparação de respostas:")
        print(result)

    elif args.command == "validate_methods":
        from controllab.analysis.stability_utils import validate_stability_methods
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        result = validate_stability_methods(tf)
        print("Validação dos métodos de estabilidade:")
        print(result)

    elif args.command == "cross_validate_poles":
        from controllab.analysis.stability_utils import cross_validate_poles
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        result = cross_validate_poles(tf)
        print("Validação cruzada dos polos:")
        print(result)

    elif args.command == "format_stability_report":
        from controllab.analysis.stability_utils import format_stability_report
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        result = format_stability_report(tf)
        print("Relatório de estabilidade formatado:")
        print(result)

    elif args.command == "stability_region_2d":
        from controllab.analysis.stability_utils import stability_region_2d
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        param1 = sp.Symbol(args.param1)
        param2 = sp.Symbol(args.param2)
        result = stability_region_2d(tf, param1, param2)
        print("Região de estabilidade 2D:")
        print(result)

    elif args.command == "root_locus_3d":
        from controllab.analysis.stability_utils import root_locus_3d
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        param1 = sp.Symbol(args.param1)
        param2 = sp.Symbol(args.param2)
        result = root_locus_3d(tf, param1, param2)
        print("Lugar das raízes 3D:")
        print(result)

    elif args.command == "analyze_sensitivity":
        from controllab.analysis.stability_utils import analyze_sensitivity
        s = sp.Symbol('s')
        expr = sp.sympify(args.ft)
        tf = SymbolicTransferFunction(sp.numer(expr), sp.denom(expr))
        param = sp.Symbol(args.param)
        result = analyze_sensitivity(tf, param)
        print("Análise de sensibilidade:")
        print(result)


    # Handler automático para funções públicas dos submódulos
    import inspect
    controllab_modules = [
        controllab.analysis,
        controllab.core,
        controllab.design,
        controllab.modeling,
        controllab.numerical
    ]
    for module in controllab_modules:
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                if args.command == name:
                    func_args = []
                    if hasattr(args, 'args') and args.args:
                        for arg in args.args:
                            try:
                                func_args.append(eval(arg))
                            except Exception:
                                func_args.append(arg)
                    result = obj(*func_args)
                    print(f'Resultado de {name}:')
                    print(result)
                    return
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
