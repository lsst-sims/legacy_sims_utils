import re

__all__ = ["SpecMap", "defaultSpecMap"]

class SpecMap(object):
    subdir_map = {'(^km)|(^kp)':'starSED/kurucz',
                  '(^bergeron)':'starSED/wDs',
                  '(^burrows)|(^(m|L|l)[0-9])':'starSED/mlt',
                  '^(Exp|Inst|Burst|Const)':'galaxySED'}
    def __init__(self, D=None):
        if D:
            self.D = D
        else:
            self.D = {}

    def __setitem__(self, key, val):
        self.D[key] = val

    def __getitem__(self, item):
        item = item.rstrip()
        if self.D.has_key(item):
            return self.D[item]
        for key, val in sorted(self.subdir_map.iteritems()):
            if re.match(key, item):
                return '{0}/{1}.gz'.format(val, item)
        raise KeyError("No path found for spectrum name: %s"%(item))

    def has_key(self, item):
        """
        Returns True if there is a map for 'item'; False if not.

        This exists primarily so that phoSim input catalog classes
        can identify columns which have no sedFilePath and then remove
        them when writing the catalog.
        """
        try:
            self.__getitem__(item)
            return True
        except:
            return False

defaultSpecMap = SpecMap(
    {'A.dat':'ssmSED/A.dat.gz',
     'Sa.dat':'ssmSED/Sa.dat.gz',
     'O.dat':'ssmSED/O.dat.gz',
     'harris_V.dat':'ssmSED/harris_V.dat.gz',
     'Cg.dat':'ssmSED/Cg.dat.gz',
     'Sv.dat':'ssmSED/Sv.dat.gz',
     'X.dat':'ssmSED/X.dat.gz',
     'K.dat':'ssmSED/K.dat.gz',
     'L.dat':'ssmSED/L.dat.gz',
     'D.dat':'ssmSED/D.dat.gz',
     'Sq.dat':'ssmSED/Sq.dat.gz',
     'C.dat':'ssmSED/C.dat.gz',
     'T.dat':'ssmSED/T.dat.gz',
     'Xk.dat':'ssmSED/Xk.dat.gz',
     'R.dat':'ssmSED/R.dat.gz',
     'Ch.dat':'ssmSED/Ch.dat.gz',
     'kurucz_sun':'ssmSED/kurucz_sun.gz',
     'Sr.dat':'ssmSED/Sr.dat.gz',
     'Cgh.dat':'ssmSED/Cgh.dat.gz',
     'Cb.dat':'ssmSED/Cb.dat.gz',
     'Xc.dat':'ssmSED/Xc.dat.gz',
     'Xe.dat':'ssmSED/Xe.dat.gz',
     'Q.dat':'ssmSED/Q.dat.gz',
     'S.dat':'ssmSED/S.dat.gz',
     'B.dat':'ssmSED/B.dat.gz',
     'V.dat':'ssmSED/V.dat.gz',
     'agn.spec':'agnSED/agn.spec.gz',
     'BD1000.dat':'starSED/gizis_SED/BD1000_interp.dat.gz',
     'BD1000e.dat':'starSED/gizis_SED/BD1000e_interp.dat.gz',
     'BD1500.dat':'starSED/gizis_SED/BD1500_interp.dat.gz',
     'BD1800.dat':'starSED/gizis_SED/BD1800_interp.dat.gz',
     'BD2000.dat':'starSED/gizis_SED/BD2000_interp.dat.gz',
     'BD325.dat':'starSED/gizis_SED/BD325_interp.dat.gz',
     'BD555.dat':'starSED/gizis_SED/BD555_interp.dat.gz',
     'PNastrom.dat':'starSED/gizis_SED/PNastrom_interp.dat.gz',
     'RedE1astrom.dat':'starSED/gizis_SED/RedE1astrom_interp.dat.gz',
     'RedE2astrom.dat':'starSED/gizis_SED/RedE2astrom_interp.dat.gz',
     'sed_flat.txt':'flatSED/sed_flat.txt.gz',
     'sed_flat_norm.txt':'flatSED/sed_flat.txt.gz'})
