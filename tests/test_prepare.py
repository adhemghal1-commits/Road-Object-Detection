from src.prepare_rdd2022 import convert_xml

XML = """<annotation><size><width>100</width><height>200</height></size>
<object><name>D40</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>100</ymax></bndbox></object>
<object><name>D44</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
</annotation>"""


def test_convert_xml(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(XML)
    lines = convert_xml(p)
    assert len(lines) == 1  # D44 skipped
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "3"
    assert (float(cx), float(cy), float(w), float(h)) == (0.3, 0.3, 0.4, 0.4)
