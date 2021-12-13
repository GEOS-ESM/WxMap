import re
import datetime as dt

class datetime(dt.datetime):

    def __init__(self, *args, **kwargs):

        super(datetime,self).__init__(*args, **kwargs)
        self.m = 'JFMAMJJASOND'

    def strftime(self, s1):

        m1 = self.month - 2
        m2 = self.month - 1
        m3 = self.month
        mmm = self.m[m1-1] + self.m[m2-1] + self.m[m3-1]

        i  = 0
        s2 = ''
        L  = len(s1)

        while i < L:
            sub = s1[i:min(i+2,L)]
            if sub == '%%':
                s2 += sub
                i += 2
            elif sub == '%3':
                s2 += mmm
                i += 2
            else:
                s2 += s1[i]
                i += 1
            
        return super(datetime,self).strftime(s2)

def fromiso(iso_string):

    dattim = re.sub('[^0-9]','', iso_string+'000000')

    year   = int(dattim[0:4])
    month  = int(dattim[4:6])
    day    = int(dattim[6:8])
    hour   = int(dattim[8:10])
    minute = int(dattim[10:12])
    second = int(dattim[12:])

    return datetime(year,month,day,hour,minute,second)

timedelta = dt.timedelta
