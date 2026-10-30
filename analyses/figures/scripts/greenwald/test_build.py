"""Check the two transformations that could silently alter figure meaning."""
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import Polygon

from analyses.figures.scripts.greenwald import build as b
from analyses.figures.common import save_panel


def test_boundary_check_uses_semantic_identity_and_actual_responder_support():
    f=pd.DataFrame([(x,y) for x in range(7) for y in range(5)],columns=['x','y'])
    f['roi_id']=np.where(f.x.eq(0),'anchor','responder')
    f['gene1']=f.x/7;f['gene2']=f.y/4
    config={'run_config':{'technology':'visium'},
            'resolved_policy':{'source_adjacency':{'method':'fixed_radius','radius_factor':1.5}}}
    bands,_,_=b.rebuild_bands(f,config,'anchor','responder')
    run={'frame':f,'names':['MES-Hyp','MES-Ast']}
    stored=pd.DataFrame([dict(surface='source_native',responder_feature_id=p,
        availability_status='finite',source_near_mass=10,source_far_mass=15,
        reachable_responder_fraction=1.,
        interface_response=f.loc[bands=='near',f'gene{i+1}'].mean()-f.loc[bands=='far',f'gene{i+1}'].mean())
        for i,p in enumerate(run['names'])])
    assert [r['program'] for r in b.verify_boundary(run,bands,stored)]==run['names']
    changed=stored.copy();changed.loc[0,'source_far_mass']=14
    with pytest.raises(ValueError,match='support counts'):b.verify_boundary(run,bands,changed)
    changed=stored.copy();changed.loc[0,'interface_response']+=.1
    with pytest.raises(ValueError,match='differs'):b.verify_boundary(run,bands,changed)
    # A disconnected responder is unavailable to the contact, not a far-band spot.
    extended=pd.concat([f,pd.DataFrame({'x':[100.],'y':[100.],
        'roi_id':['responder'],'gene1':[.2],'gene2':[.3]})],ignore_index=True)
    labels,hops,_=b.rebuild_bands(extended,config,'anchor','responder')
    assert labels[-1]=='unreachable' and hops[-1]<0
    np.testing.assert_array_equal(labels[:-1],bands)


def test_source_and_consensus_maps_preserve_values_and_share_display_scale(tmp_path):
    f=pd.DataFrame({'x':[0.,1.,2.],'y':[0.,1.,0.],'roi_id':['1']*3,
                    'gene1':[-.5,.2,1.5],'gene2':[.4,.5,.8]})
    poly=Polygon([(-.3,-.3),(.3,-.3),(.3,.3),(-.3,.3)],
                 holes=[[(-.1,-.1),(-.1,.1),(.1,.1),(.1,-.1)]])
    run={'frame':f,'names':['AC','Reactive-Ast'],'spacing':1.,'limits':{'AC':1.5},
         'outlines':{'1':Polygon([(-.4,-.4),(2.4,-.4),(2.4,1.4),(-.4,1.4)])},
         'hexes':pd.DataFrame({'hex_geometry_wkt':[poly.wkt], 'gene1_density_ensemble':[.37]})}
    evidence=SimpleNamespace(run=lambda *_:run,map_settings=[])
    for surface,expected in [('S',f.gene1.to_numpy()),('CE',np.array([.37]))]:
        fig=plt.figure(figsize=(3,3))
        ax=b.map_draw(evidence,'example','AC_Reactive-Ast','AC',surface,['1'])(fig,(0,0,1,1))
        artist=ax.collections[0 if surface=='S' else 1]
        np.testing.assert_array_equal(artist.get_array(),expected)
        assert artist.get_clim()==(-1.5,1.5)
        if surface=='CE':assert list(artist.get_paths()[0].codes).count(1)==2
        assert 'AC' in [t.get_text() for t in fig.texts]
        stem=tmp_path/surface
        save_panel(fig,stem.parent,stem.name)
        ns='{http://www.w3.org/2000/svg}'
        svg=ET.parse(stem.with_suffix('.svg')).getroot()
        canvas=svg.find(f".//{ns}g[@id='patch_1']/{ns}path")
        assert canvas.get('style')=='fill: none'
        # White label backings remain; only canvas/axes become transparent.
        assert any('fill: #ffffff' in p.get('style','') for p in svg.iter(ns+'path'))
        with Image.open(stem.with_suffix('.png')) as preview:
            assert preview.convert('RGB').getpixel((0,0))==(255,255,255)
