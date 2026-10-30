"""A4 review layouts; individual vector panels remain available for final assembly."""
import matplotlib.pyplot as plt
from analyses.figures.common import save_panel

DISPLAY_PANELS = {
    'FigureS7': ('A_GBM_location', 'B1_GBM_Stemlike_consensus', 'B2_GBM_Interferon_consensus',
                'C1_GBM_Stemlike_ensemble_detail', 'C2_GBM_Interferon_ensemble_detail', 'D_GBM_enrichment'),
    'FigureS8': ('A1_Breast_ACTA2_cells', 'A2_Breast_ACTA2_ensemble',
                'A3_Breast_FASN_cells', 'A4_Breast_FASN_ensemble', 'B_Breast_enrichment'),
}


def build_preview(directory, name):
    figure = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    if name == 'FigureS7':
        title = 'Figure S7 | Stem-like and interferon-related programs in GBM'
        placements = [
            ('A_GBM_location', (.04,.72,.38,.245), 'A'),
            ('D_GBM_enrichment', (.44,.735,.53,.23), 'D'),
            ('B1_GBM_Stemlike_consensus', (.035,.36,.455,.365), 'B'),
            ('B2_GBM_Interferon_consensus', (.51,.36,.455,.365), ''),
            *[(panel, (.135+.48*index,.095,.25,.255), 'C' if index == 0 else '')
              for index,panel in enumerate(DISPLAY_PANELS[name][3:5])],
        ]
        note = ('P28_post: local tissue context and complete interface and lymphoid components.\n'
                'Consensus enlargements retain the main maps’ colour limits; enrichment uses whole ROIs.\n'
                'Additional full-field cell maps and all individual vector panels accompany this figure.')
    else:
        title = 'Figure S8 | ACTA2 and FASN in breast Xenium tissue'
        placements = [
            (panel, (.065+.45*(index%2), .565-.35*(index//2), .42,.345), 'A' if index == 0 else '')
            for index,panel in enumerate(DISPLAY_PANELS[name][:4])
        ] + [('B_Breast_enrichment', (.21,.02,.60,.205), 'B')]
        note = ''
    figure.text(.04,.98,title, fontsize=12, fontweight='bold', va='top')
    for panel, rectangle, letter in placements:
        axis = figure.add_axes(rectangle)
        axis.imshow(plt.imread(directory/'panels'/f'{panel}.png'))
        axis.set_axis_off()
        if letter:
            figure.text(rectangle[0], rectangle[1]+rectangle[3], letter, fontsize=12,
                        fontweight='bold', va='top')
    if note:
        figure.text(.06,.065,note,fontsize=8,va='top',linespacing=1.5)
    save_panel(figure, directory, f'{name}_layout_preview')
