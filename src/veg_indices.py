'''Vegetation indices class'''
import numpy as np

class VI(object):
    ''' example usage: VI(wave_comb,840,668).ND(preprocessed_image), for preprocessed_image(ch,x,y):'''
    def __init__(self,full_wave,wave1,wave2,wave3=None,wave4=None):
        self.full_wave = full_wave
        self.wave1 = wave1
        self.wave2 = wave2
        self.wave3 = wave3
        self.i1 = self.f_n_i(full_wave,wave1)
        self.i2 = self.f_n_i(full_wave,wave2)
        if wave3:
            self.i3 = self.f_n_i(full_wave,wave3)
        if wave4:
            self.i4 = self.f_n_i(full_wave,wave4)

    def f_n_i(self,array, value):
        '''https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array'''
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def ratio(self,R):
        '''For ratio'''
        return R[self.i1]/R[self.i2]

    def ND(self,R):
        '''For normalised difference'''
        return (R[self.i1]-R[self.i2])/(R[self.i1]+R[self.i2])
    
    def DIRT(self,R,beta):
        '''Weed detection dirt'''
        return np.sign(beta-R[self.i2])*self.ND(R)
    
    def RD(self,R):
        return (R[self.i1]-R[self.i2])/np.sqrt(R[self.i1]+R[self.i2])
    
    def RD3(self,R):
        '''For LCI (linear combination index)'''
        return (R[self.i1]-R[self.i2])/np.sqrt(R[self.i1]+R[self.i3])
    
    def SA(self,R,l):
        '''For soil adjusted VI'''
        return (1+l)*(R[self.i1]-R[self.i2])/(R[self.i1]+R[self.i2]+l)
    
    def MSA(self,R):
        '''For modified soil adjusted index'''
        return (2*R[self.i1]+1-np.sqrt((2*R[self.i1]+1)**2-8*(R[self.i1]-R[self.i2])))/2

    def PSR(self,R):
        '''For plant senescence reflectance red, green, red-edge'''
        return (R[self.i1]-R[self.i2])/(R[self.i3])
    
    def MCAR(self,R):
        '''For modified chlorophyll and reflectance index,700nm, 670nm, 550nm'''
        return R[self.i1]/R[self.i2]*(R[self.i1]-R[self.i2]-0.2*(R[self.i1]-R[self.i3]))
    
    def CAR(self,R,a,b):
        '''For chlorophyll absorption ratio'''
        return (R[self.i1]*(a*R[self.i2]+b*R[self.i2]))/(R[self.i2]*np.sqrt(a**2+1))

    def TVI(self,R,a=120,b=200):
        '''For Triangular Vegetation Index, 750, 670, 550'''
        return 0.5*(a*(R[self.i1]-R[self.i3])-b*(R[self.i2]-R[self.i3]))
    
    def SPVI(self,R,a=3.7,b=1.2):
        '''For Spectral Polygon Vegetation Index, 800, 670, 530'''
        return 0.4*(a*(R[self.i1]-R[self.i2])-b*(R[self.i3]-R[self.i2]))
    
    def MTVI1(self,R):
        '''For modified TVI, 800, 670, 550'''
        return 1.2*(1.2*(R[self.i1]-R[self.i3])-2.5*(R[self.i2]-R[self.i3]))

    def TCARI(self,R):
        '''For Transformed chlorophyll absorption ratio index, 700, 670, 550'''
        return 3*((R[self.i1]-R[self.i2])-0.2*(R[self.i1]-R[self.i3])*(R[self.i1]/R[self.i2]))

    def MCARI(self,R):
        '''For modified chlorophyll absorption ratio index, 700, 670, 550'''
        return ((R[self.i1]-R[self.i2])-0.2*(R[self.i1]-R[self.i3]))*(R[self.i1]/R[self.i2])
    
    def MCARI1(self,R):
        '''For modified CARI1 800, 670, 550'''
        return 1.2*(2.5*(R[self.i1]-R[self.i2])-1.3*(R[self.i1]-R[self.i3]))
    
    def MCARI2(self,R):
        '''For modified CARI2 800, 670, 550'''
        return 1.5*(2.5*(R[self.i1]-R[self.i2])-1.3*(R[self.i1]-R[self.i3]))/np.sqrt((2*R[self.i1]+1)**2-(6*R[self.i1]-5*np.sqrt(R[self.i2]))-0.5)
    
    def MTVI2(self,R):
        '''For modified TVI reduced soil contamination 800, 670, 550'''
        return 1.5*(1.2*(R[self.i1]-R[self.i3])-2.5*(R[self.i2]-R[self.i3]))/np.sqrt((2*R[self.i1]+1)**2-(6*R[self.i1]-5*np.sqrt(R[self.i2]))-0.5)

    def VDVI(self,R):
        '''For visible band difference vegetation index, red, green, blue'''
        return (2*R[self.i2]-R[self.i1]-R[self.i3])/(2*R[self.i2]+R[self.i1]+R[self.i3])
    
    def Mahlein3idx(self,R,const):
        '''This is a combination of healthy and CLS index as reported from Mahlein et al. (2013)
            for CLSI its 698, 570, 734 and const = 1'''
        return self.ND(R)-const*R[self.i3] 
    
    def GVI(self,R):
        '''Green vegetation index, green, red, rededge, NIR'''
        return -0.283*R[self.i1]-0.66*R[self.i2]+0.577*R[self.i3]+0.388*R[self.i4]
    
    def ExG(self,R):
        '''Excess green index, green, red, blue'''
        return 2*R[self.i1]-R[self.i2]-R[self.i3]
    
    def ExR(self,R):
        '''Excess red index, red, green'''
        return 1.4*R[self.i1]+R[self.i2]
    
    def VEG(self,R):
        '''Vegetative index, green, red, blue'''
        return R[self.i1]/(R[self.i2]**0.667*R[self.i3]**0.333)
    
    def WoI(self,R):
        '''Woebbecke index, green, red ,blue'''
        return (R[self.i1]-R[self.i3])/(R[self.i2]-R[self.i1])
    
    def CIVE(self,R):
        '''color index of vegetation, green, red, blue'''
        return 0.441*R[self.i2]-0.881*R[self.i1]+0.385*R[self.i3]+18.78745
    
    def MCARIOSAVI(self,R):
        '''MCARI and OSAVI from Barreto et al. NIR, Red and Red Edge'''
        return (R[self.i3]/R[self.i2]*(R[self.i3]-R[self.i2]-0.2*(R[self.i3]-R[self.i2])))/self.SA(0.16,R)
    
    def EVI2(self,R):
        '''Enhanced vegetation index NIR, red'''
        return 2.5*(R[self.i1]-R[self.i2])/(R[self.i1]+2.4*R[self.i2]+1)
    
    def TDVI(self,R):
        '''transformed difference vegetation index, NIR, red'''
        return 1.5*(R[self.i1]-R[self.i2])/np.sqrt(R[self.i1]**2+R[self.i2]+0.5)
    
    def WDRVI(self,R):
        '''wide-range dynamic vegetation index NIR, red'''
        return (0.05*R[self.i1]-R[self.i2])/(0.05*R[self.i1]+R[self.i2])

    def MSR(self,R):
        '''MSR (Modified simple ratio)'''
        return (R[self.i1]/R[self.i2]-1)/np.sqrt(R[self.i1]/R[self.i3]+1)
    
    def RVSI(self,R):
        '''Red-Edge Vegetation Stress Index, 752, 732, 712'''
        return (R[self.i3]+R[self.i1])/2-R[self.i2]
