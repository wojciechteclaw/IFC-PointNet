from collections import namedtuple

IfcGeometryEntity = namedtuple('IfcGeometryEntity', ['faces', 'guid', 'location', 'rotation', 'valid', 'vertices'])