import numpy as np

constGo = 32.1740485564304 #From ICAO 7488 go=9.80665, standard acceleration due to gravity at latitude 45 deg, 32 min, 33 sec using Lambert's equation of the acceleration due to gravity function latitude (PER ICAO 7488 SEE: U.S. Committee Extension Standard Atmosphere: U.S. Standard Atmosphere, 1962. U.S. Government) 

# Conversion Factors
constConvLbfPerInch2ToLbfPerFt2 = 144 #lb/in2 to lb/ft2 (exact)
constConvKelvinToRankine = 1.8 #Kelvin to Rankine (exact)
constConvFtToMeter = 0.3048 #feet to meters (exact)
constConvNmToMeter = 1852 #nm to meters (exact)
constConvHrToSec = 3600 #hour to seconds (exact)
constConvMileToFt = 5280 #statute miles to feet (exact)
constConvFtToInch = 12 #feet to inch (exact)
constConvLbfToNewton = 4.4482216152605 #lb force to Newton
constConvHorsepowerToLbfFtPerSec = 550 #horsepower to lbf-ft/sec
constConvFtPerSecToKts = 0.592483801295896 #ft/sec to knots, derived from 1852 m/nm, 3600 sec/hr & 0.3048 m/ft
constConvLbmToKg = 0.45359237 #lb mass to kilogram, based on National Bureau of Standards 5348, 1959 (exact)
constConvLbmToOunce = 16 #lb mass to ounces (exact)
constConvShortTonToLbm = 2000 #short ton (i.e., US "ton") to lb mass (exact)
constConvInHgToHectoPascal = 33.86389 #hPa to inHg, based on NIST Special Pub 811, 2008

#Temperatures for zero deg C and zero deg F on absolute temperature scales
constZeroDegCelsiusInKelvin = 273.15 #Temperature in Kelvin at zero Celsius, K
constZeroDegFahrenheitInRankine = 459.67 #Temperature in Rankine at zero Fahrenheit, R

#Physical constants defining the standard atmosphere
constAo = 661.478594435162 #Derived from From ICAO 7488 values for To, R. Uses 1852 m/nm. Speed of sound at sea level, knots
constGammaAir = 1.4 #From ICAO 7488, ratio of specific heats for air
constRAir = 3089.81137753942 #From ICAO 7488, R=287.05287 m2/(s2 K) converted with 0.3048 m/ft, gas constant, dry air, ft2/s2/K
constRadiusEarth = 20855531.496063 #From ICAO 7488, r=6356766 m, 0.3048 m/ft, nominal radius earth, feet

#Geopotential altitudes (in feet) at breaks in temperature profile
constTropopause11kmInFt = 36089.2388451444 #From ICAO 7488, 11000 m / 0.3048 m/ft
constTopIsoThermLayerStrat20kmInFt = 65616.7979002625 #From ICAO 7488, 20000 m / 0.3048 m/ft
constTop1stInverLayerStrat32kmInFt = 104986.87664042 #From ICAO 7488, 32000 m / 0.3048 m/ft
constStratopauseStart47kmInFt = 154199.475065617 #From ICAO 7488, 47000 m / 0.3048 m/ft
constStratopauseEnd51kmInFt = 167322.83464567 #From ICAO 7488, 51000 m / 0.3048 m/ft

#Pre-computed atmospheric pressure ratios (deltas) at breaks in temperature profile
constDeltaTropopause = 0.223360869430129
constDeltaStrat20km = 0.054032839124412
constDeltaStrat32km = 8.5666496582306E-03
constDeltaStrat47km = 1.09455488149331E-03

#Lapse rates to top of stratosphere
constLapseTrop = -0.0019812 #From ICAO 7488 standard lapse rate in troposhere, -6.50°C/(1000 m) & 0.3048 m/ft
#constLStrat1 would be zero (lower stratosphere is isothermal) - not needed
constLapseStrat2 = 0.0003048 #From ICAO 7488, lapse rate from 20 km to 32 km +1.00°C/(1000 m) & 0.3048 m/ft
constLapseStrat3 = 0.00085344 #From ICAO 7488, lapse rate from 32 km to 47 km +2.80°C/(1000 m) & 0.3048 m/ft

#Atmospheric temperature (in Kelvin) at breaks in temperature profile
constOatIsoLayerStrat11to20kmInK = 216.65 #From ICAO 7488, OAT in isothermal layer of lower stratosphere, K
constOatStrat32kmInK = 228.65 #From ICAO 7488, OAT at transition between first and second inversion layer, K
constOatStratopause47to51kmInK = 270.65 #From ICAO 7488, OAT in stratopause (47 to 51 km), K

#Characteristics at sea level
constTo = 288.15 #From ICAO 7488, sea level value of absolute temperature in standard atmosphere, K (518.67 R)
constAo = 661.478594435162 #Derived from From ICAO 7488 values for To, R. Uses 1852 m/nm. Speed of sound at sea level, knots
constPo = 2116.21662367394 #From ICAO 7488 Po=101325 N/m2, converted by 0.3048 m/ft & 4.4482216152605 N/lbf
constRhoo = 2.37689240667515E-03 #From ICAO 7488, rhoSL=l.225 kg/m3, 4.4482216152605 (kg m)/(lbf s2) & 1 slug = 1 (s2 lbf)/ft

#Program execution constants
constMaxIter = 100 #Maximum iterations to converge solution (i.e., supersonic Mach = f(hp, kcas))
constMachEpsilon = 0.000001 #Arbitrary epsilon for Mach convergence at supersonic speeds
constKcasEpsilon = 0.0001 #Arbitrary epsilon for KCAS convergence at supersonic speeds
constAeroErr = -99999999.9999999 #Initialization value, a large-magnitude, oddball, negative value

def ConvKelvinToCelsius(oatKelvin): 
	return oatKelvin - constZeroDegCelsiusInKelvin
	
def ConvKelvinToRankine(oatKelvin):
	return oatKelvin * constConvKelvinToRankine 

def ConvKelvinToFahrenheit(oatKelvin):
	return ConvKelvinToRankine(oatKelvin) - constZeroDegFahrenheitInRankine
	
def ConvCelsiusToKelvin(oatCelsius):
	return oatCelsius + constZeroDegCelsiusInKelvin 

def ConvCelsiusToRankine(oatCelsius):
	return ConvKelvinToRankine(ConvCelsiusToKelvin(oatCelsius))

def ConvCelsiusToFahrenheit(oatCelsius):
	return ConvCelsiusToRankine(oatCelsius) - constZeroDegFahrenheitInRankine 

def ConvRankineToKelvin(oatRankine):
	return oatRankine / constConvKelvinToRankine
	
def ConvRankineToCelsius(oatRankine):
	return ConvRankineToKelvin(oatRankine) - constZeroDegCelsiusInKelvin
	
def ConvRankineToFahrenheit(oatRankine):
	return oatRankine - constZeroDegFahrenheitInRankine

def ConvFahrenheitToKelvin(oatFahrenheit):
	return ConvFahrenheitToRankine(oatFahrenheit) / constConvKelvinToRankine

def ConvFahrenheitToCelsius(oatFahrenheit): 
	return (oatFahrenheit - 32) / constConvKelvinToRankine 

def ConvFahrenheitToRankine(oatFahrenheit): 
	return oatFahrenheit + constZeroDegFahrenheitInRankine 
	
def ConvFtPerSecToKts(ftPerSec):
	return ftPerSec * constConvFtPerSecToKts 

def ConvFtPerSecToMilePerHr(ftPerSec): 
	return ftPerSec * constConvHrToSec / constConvMileToFt

def ConvFtPerSecToMeterPerSec(ftPerSec):
	return ftPerSec * constConvFtToMeter 

def ConvFtPerSecToKmPerHr(ftPerSec): 
	return ftPerSec * constConvHrToSec * constConvFtToMeter / 1000 

def ConvKtsToFtPerSec(kts):
	return kts / constConvFtPerSecToKts 
 
def ConvKtsToMilePerHr(kts):
	return kts / constConvFtPerSecToKts / constConvMileToFt * constConvHrToSec 

def ConvKtsToMeterPerSec(kts):
	return kts / constConvFtPerSecToKts * constConvFtToMeter 

def ConvKtsToKmPerHr(kts):
	return kts * constConvNmToMeter / 1000 

def ConvMilePerHrToFtPerSec(milePerHr):
	return milePerHr * constConvMileToFt / constConvHrToSec

def ConvMilePerHrToKts(milePerHr): 
	return milePerHr * constConvMileToFt / constConvHrToSec * constConvFtPerSecToKts

def ConvMilePerHrToMeterPerSec(milePerHr): 
	return milePerHr * constConvMileToFt * constConvFtToMeter / constConvHrToSec

def ConvMilePerHrToKmPerHr(milePerHr):
	return milePerHr * constConvMileToFt * constConvFtToMeter / 1000 
 
def ConvMeterPerSecToFtPerSec(mPerSec):
	return mPerSec / constConvFtToMeter 
	
def ConvMeterPerSecToKts(mPerSec):
	return mPerSec * constConvFtPerSecToKts / constConvFtToMeter

def ConvMeterPerSecToMilePerHr(mPerSec):
	return mPerSec / constConvFtToMeter / constConvMileToFt * constConvHrToSec
	
def ConvMeterPerSecToKmPerHr(mPerSec):
	return mPerSec * constConvHrToSec / 1000 

def ConvKmPerHrToFtPerSec(kmPerHr): 
	return kmPerHr * 1000 / constConvFtToMeter / constConvHrToSec

def ConvKmPerHrToKts(kmPerHr):
	return kmPerHr * 1000 / constConvNmToMeter 

def ConvKmPerHrToMilePerHr(kmPerHr): 
	return kmPerHr * 1000 / constConvFtToMeter / constConvMileToFt 

def ConvKmPerHrToMeterPerSec(kmPerHr): 
	return kmPerHr * 1000 / constConvHrToSec 

def ConvSlugPerFt3ToKgPerMeter3(density):
	return density * constConvLbfToNewton / constConvFtToMeter ** 4
 
def ConvKgPerMeter3ToSlugPerFt3(density): 
	return density / constConvLbfToNewton * constConvFtToMeter ** 4 

def ConvLbfPerFt2ToLbfPerInch2(lbfPerFt2):
	return lbfPerFt2 / constConvLbfPerInch2ToLbfPerFt2

def ConvLbfPerFt2ToInHg(lbfPerFt2):
	return ConvLbfPerFt2ToHPa(lbfPerFt2) / constConvInHgToHectoPascal

def ConvLbfPerFt2ToHPa(lbfPerFt2):
	return lbfPerFt2 / constConvFtToMeter ** 2 * constConvLbfToNewton / 100
	
def ConvLbfPerInch2ToLbfPerFt2(lbfPerInch2): 
	return lbfPerInch2 * constConvLbfPerInch2ToLbfPerFt2

def ConvLbfPerInch2ToInHg(lbfPerInch2):
	return ConvLbfPerInch2ToHPa(lbfPerInch2) / constConvInHgToHectoPascal 

def ConvLbfPerInch2ToHPa(lbfPerInch2):
	return ConvLbfPerFt2ToHPa(lbfPerInch2 * constConvLbfPerInch2ToLbfPerFt2) 

def ConvInHgToLbfPerFt2(inHg):
	return ConvHPaToLbfPerFt2(ConvInHgToHPa(inHg)) 

def ConvInHgToLbfPerInch2(inHg):
	return ConvHPaToLbfPerInch2((ConvInHgToHPa(inHg))) 

def ConvInHgToHPa(inHg):
	return inHg * constConvInHgToHectoPascal 

def ConvHPaToLbfPerFt2(hPa):
	return hPa * constConvFtToMeter ** 2 / constConvLbfToNewton * 100 

def ConvHPaToLbfPerInch2(hPa):
	return ConvHPaToLbfPerFt2(hPa) / constConvLbfPerInch2ToLbfPerFt2 

def ConvHPaToInHg(hPa):
	return hPa / constConvInHgToHectoPascal 

def ConvDegToRad(deg):
	return deg * np.pi / 180

def ConvRadToDeg(rad):
	return rad / np.pi * 180
	
def ConvFtPerNmToPctGrad(ftPerNm):
	return ConvFtToNm(ftPerNm) * 100	
	
def ConvFtPerNmToDeg(ftPerNm):
	return ConvRadToDeg(np.atan(ConvFtToNm(ftPerNm))) 

def ConvPctGradToFtPerNm(pct):
	return ConvNmToFt(pct / 100) 

def ConvPctGradToDeg(pct):
	return ConvRadToDeg(np.atan(pct / 100)) 

def ConvDegToFtPerNm(deg):
	return ConvNmToFt(np.tan(ConvDegToRad(deg))) 

def ConvDegToPctGrad(deg):
	return np.tan(ConvDegToRad(deg)) * 100 

def ConvFtToNm(ft):
	return ft * constConvFtToMeter / constConvNmToMeter 

def ConvFtToMile(ft):
	return ft / constConvMileToFt 

def ConvFtToMeter(ft): 
	return ft * constConvFtToMeter 

def ConvFtToKm(ft):
	return ConvFtToMeter(ft) / 1000 

def ConvNmToFt(nm):
	return nm * constConvNmToMeter / constConvFtToMeter
	
def ConvNmToMile(nm):
	return ConvNmToFt(nm) / constConvMileToFt

def ConvNmToMeter(nm):
	return nm * constConvNmToMeter
	
def ConvNmToKm(nm):
	return ConvNmToMeter(nm) / 1000
	
def ConvMileToFt(mile):
	return mile * constConvMileToFt
	
def ConvMileToNm(mile):
	return ConvFtToNm(ConvMileToFt(mile))
	
def ConvMileToMeter(mile):
	return ConvFtToMeter(ConvMileToFt(mile))
	
def ConvMileToKm(mile):
	return ConvFtToKm(ConvMileToFt(mile))

def ConvMeterToFt(meter):
	return meter / constConvFtToMeter

def ConvMeterToNm(meter):
	return meter / constConvNmToMeter
	
def ConvMeterToMile(meter):
	return meter / constConvFtToMeter / constConvMileToFt
	
def ConvMeterToKm(meter):
	return meter / 1000
	
def ConvKmToFt(km):
	return ConvMeterToFt(ConvKmToMeter(km))
	
def ConvKmToNm(km):
	return ConvMeterToNm(ConvKmToMeter(km))
	
def ConvKmToMile(km):
	return ConvMeterToMile(ConvKmToMeter(km))

def ConvKmToMeter(km):
	return km * 1000
	
def ConvLbfToNewton(lbf):
	return lbf * constConvLbfToNewton
	
def ConvLbfToKiloNewton(lbf):
	return ConvNewtonToKiloNewton(ConvLbfToNewton(lbf)) 

def ConvNewtonToLbf(newton):
	return newton / constConvLbfToNewton

def ConvNewtonToKiloNewton(newton):
	return newton / 1000
	
def ConvKiloNewtonToLbf(kilonewton):
	return ConvNewtonToLbf(ConvKiloNewtonToNewton(kilonewton))

def ConvKiloNewtonToNewton(kilonewton):
	return kilonewton * 1000

def ConvLbfFtToLbfInch(lbfFt):
	return lbfFt * constConvFtToInch 
 
def ConvLbfFtToNewtonMeter(lbfFt):
	return ConvFtToMeter(ConvLbfToNewton(lbfFt)) 

def ConvLbfInchToLbfFt(lbfInch):
	return lbfInch / constConvFtToInch
	
def ConvLbfInchToNewtonMeter(lbfInch):
	return ConvLbfFtToNewtonMeter(ConvLbfInchToLbfFt(lbfInch))

def ConvNewtonMeterToLbfFt(newtonMeter): 
	return ConvMeterToFt(ConvNewtonToLbf(newtonMeter)) 

def ConvNewtonMeterToLbfInch(newtonMeter):
	return ConvLbfFtToLbfInch(ConvNewtonMeterToLbfFt(newtonMeter)) 
	
def ConvFtLbfPerSecToHorsepower(ftLbfPerSec):
	return ftLbfPerSec / constConvHorsepowerToLbfFtPerSec                        
	
def ConvFtLbfPerSecToWatt(ftLbfPerSec):
	return ConvLbfFtToNewtonMeter(ftLbfPerSec)
	
def ConvFtLbfPerSecToKilowatt(ftLbfPerSec):
	return ConvFtLbfPerSecToWatt(ftLbfPerSec) / 1000
	
def ConvHorsepowerToFtLbfPerSec(horsepower):
	return horsepower * constConvHorsepowerToLbfFtPerSec
	
def ConvHorsepowerToWatt(horsepower):
	return ConvFtLbfPerSecToWatt(ConvHorsepowerToFtLbfPerSec(horsepower))
	
def ConvHorsepowerToKilowatt(horsepower):
	return ConvHorsepowerToWatt(horsepower) / 1000
	
def ConvWattToFtLbfPerSec(watt):
	return ConvNewtonMeterToLbfFt(watt)

def ConvWattToHorsepower(watt):
	return ConvFtLbfPerSecToHorsepower(ConvWattToFtLbfPerSec(watt))
	
def ConvWattToKilowatt(watt):
	return watt / 1000
	
def ConvKilowattToFtLbfPerSec(kilowatt): 
	return ConvWattToFtLbfPerSec(kilowatt * 1000)
	
def ConvKilowattToHorsepower(kilowatt):
	return ConvWattToHorsepower(kilowatt * 1000) 

def ConvKilowattToWatt(kilowatt):
	return kilowatt * 1000 

def ConvLbmToSlug(lbm):
	return lbm / constGo 

def ConvLbmToOunce(lbm):
	return lbm * constConvLbmToOunce 

def ConvLbmToUsTon(lbm):
	return lbm / constConvShortTonToLbm
	
def ConvLbmToGram(lbm): 
	return ConvKgToGram(ConvLbmToKg(lbm)) 

def ConvLbmToKg(lbm):
	return lbm * constConvLbmToKg
	
def ConvLbmToMetricTon(lbm):
	return ConvKgToMetricTon(ConvLbmToKg(lbm)) 

def ConvSlugToLbm(slug): 
	return slug * constGo 

def ConvSlugToOunce(slug):
	return ConvLbmToOunce(ConvSlugToLbm(slug)) 

def ConvSlugToUsTon(slug):
	return ConvLbmToUsTon(ConvSlugToLbm(slug)) 
	
def ConvSlugToGram(slug): 
	return ConvLbmToGram(ConvSlugToLbm(slug)) 

def ConvSlugToKg(slug):
	return ConvLbmToKg(ConvSlugToLbm(slug)) 

def ConvSlugToMetricTon(slug):
	return ConvLbmToMetricTon(ConvSlugToLbm(slug)) 

def ConvOunceToLbm(ounce):
	return ounce / constConvLbmToOunce
	
def ConvOunceToSlug(ounce):
	return ConvLbmToSlug(ConvOunceToLbm(ounce)) 

def ConvOunceToUsTon(ounce):
	return ConvLbmToUsTon(ConvOunceToLbm(ounce)) 

def ConvOunceToGram(ounce):
	return ConvLbmToGram(ConvOunceToLbm(ounce)) 

def ConvOunceToKg(ounce):
	return ConvLbmToKg(ConvOunceToLbm(ounce)) 

def ConvOunceToMetricTon(ounce):
	return ConvLbmToMetricTon(ConvOunceToLbm(ounce))
	
def ConvUsTonToLbm(usTon):
	return usTon * constConvShortTonToLbm 

def ConvUsTonToSlug(usTon):
	return ConvLbmToSlug(ConvUsTonToLbm(usTon)) 

def ConvUsTonToOunce(usTon):
	return ConvLbmToOunce(ConvUsTonToLbm(usTon)) 

def ConvUsTonToGram(usTon): 
	return ConvLbmToGram(ConvUsTonToLbm(usTon))

def ConvUsTonToKg(usTon):
	return ConvLbmToKg(ConvUsTonToLbm(usTon)) 
 
def ConvUsTonToMetricTon(usTon):
	return ConvLbmToMetricTon(ConvUsTonToLbm(usTon)) 
 
def ConvGramToLbm(gram):
	return ConvKgToLbm(ConvGramToKg(gram)) 

def ConvGramToSlug(gram): 
	return ConvKgToSlug(ConvGramToKg(gram)) 

def ConvGramToOunce(gram):
	return ConvKgToOunce(ConvGramToKg(gram))

def ConvGramToUsTon(gram):
	return ConvKgToUsTon(ConvGramToKg(gram))

def ConvGramToKg(gram): 
	return gram / 1000 

def ConvGramToMetricTon(gram):
	return ConvKgToMetricTon(ConvGramToKg(gram)) 

def ConvKgToLbm(kg):
	return kg / constConvLbmToKg 

def ConvKgToSlug(kg):
	return ConvLbmToSlug(ConvKgToLbm(kg)) 
	
def ConvKgToOunce(kg):
	return ConvLbmToOunce(ConvKgToLbm(kg)) 

def ConvKgToUsTon(kg):
	return ConvLbmToUsTon(ConvKgToLbm(kg)) 

def ConvKgToGram(kg):
	return kg * 1000 
	
def ConvKgToMetricTon(kg):
	return kg / 1000 

def ConvMetricTonToLbm(metricTon):
	return ConvKgToLbm(ConvMetricTonToKg(metricTon)) 

def ConvMetricTonToSlug(metricTon): 
	return ConvLbmToSlug(ConvMetricTonToLbm(metricTon))

def ConvMetricTonToOunce(metricTon):
	return ConvLbmToOunce(ConvMetricTonToLbm(metricTon))

def ConvMetricTonToUsTon(metricTon):
	return ConvLbmToUsTon(ConvMetricTonToLbm(metricTon))
	
def ConvMetricTonToGram(metricTon):
	return ConvKgToGram(ConvMetricTonToKg(metricTon))

def ConvMetricTonToKg(metricTon):
	return metricTon * 1000 
	
def Kcas_fHpMach(hp, mach):
    qc = Qc_lbfPerFt2_fHpMach(hp, mach)
    return Kcas_fQc(qc)

def Kcas_fQc(qc):
    kcas = constAo * np.sqrt(2 / (constGammaAir - 1) * (np.power(qc / constPo + 1, (constGammaAir - 1) / constGammaAir) - 1))
    if (kcas > constAo):
        count = 0 #Counter included as safety mechanism to prevent infinite loop in event of unconverged solution
        kcasLast = kcas
        firstConstCoeff = np.power((constGammaAir + 1) / 2, 0.5 * constGammaAir / (1 - constGammaAir)) * np.power((constGammaAir + 1) / 2 / constGammaAir, 0.5 / (1 - constGammaAir))
        secondConstCoeff = 2 * constGammaAir / (constGammaAir - 1)
        while True:
            kcasLast = kcas
            kcas = constAo * firstConstCoeff * np.sqrt((qc / constPo + 1) * np.power(1 - 1 / (secondConstCoeff * (kcasLast / constAo)**2), 1 / (constGammaAir - 1)))
            count = count + 1
            if (count > constMaxIter):
                #Set kcas and kcasLast equal to constAeroErr; terminates loop and returns error-flag value
                kcas = constAeroErr
                kcasLast = constAeroErr
                break
            if (abs(kcas - kcasLast) < constKcasEpsilon):
                break
    return kcas

def Kcas_fHpKtasOatKelvin(hp, ktas, oatKelvin):
    mach = Mach_fKtasOatKelvin(ktas, oatKelvin)
    return Kcas_fHpMach(hp, mach)

def Kcas_fHpKtasOatCelsius(hp, ktas, oatCelsius):
    mach = Mach_fKtasOatCelsius(ktas, oatCelsius)
    return Kcas_fHpMach(hp, mach)    

def Kcas_fHpKtasOatRankine(hp, ktas, oatRankine):
    mach = Mach_fKtasOatRankine(ktas, oatRankine)
    return Kcas_fHpMach(hp, mach)

def Kcas_fHpKtasOatFahrenheit(hp, ktas, oatFahrenheit):
    mach = Mach_fKtasOatFahrenheit(ktas, oatFahrenheit)
    return Kcas_fHpMach(hp, mach)

def Kcas_fHpKtasIsaDevCelsius(hp, ktas, isaDevCelsius):
    mach = Mach_fHpKtasIsaDevCelsius(hp, ktas, isaDevCelsius)
    return Kcas_fHpMach(hp, mach)

def Kcas_fHpKtasIsaDevFahrenheit(hp, ktas, isaDevFahrenheit):
    mach = Mach_fHpKtasIsaDevFahrenheit(hp, ktas, isaDevFahrenheit)
    return Kcas_fHpMach(hp, mach)    

def Qc_lbfPerFt2_fKcas(kcas):
    if kcas < constAo:
        Qc_lbfPerFt2 = constPo * (np.power(1 + (constGammaAir - 1) / 2 * (kcas / constAo)**2, constGammaAir / (constGammaAir - 1)) - 1)
    else:
        Qc_lbfPerFt2 = constPo * (np.power((constGammaAir + 1) / 2 * (kcas / constAo)**2, constGammaAir / (constGammaAir - 1)) * np.power((constGammaAir + 1) / (1 - constGammaAir + 2 * constGammaAir * (kcas / constAo)**2), 1 / (constGammaAir - 1)) - 1)
    return Qc_lbfPerFt2

def Qc_lbfPerFt2_fHpMach(hp, mach):
    if mach < 1:
        Qc_lbfPerFt2 = Pstatic_lbfPerFt2_fHp(hp) * (np.power(1 + (constGammaAir - 1) / 2 * mach**2, constGammaAir / (constGammaAir - 1)) - 1)
    else:
        Qc_lbfPerFt2 = Pstatic_lbfPerFt2_fHp(hp) * (np.power((constGammaAir + 1) / 2 * mach**2, constGammaAir / (constGammaAir - 1)) * np.power((constGammaAir + 1) / (1 - constGammaAir + 2 * constGammaAir * mach**2), 1 / (constGammaAir - 1)) - 1)
    return Qc_lbfPerFt2

def Pstatic_lbfPerFt2_fHp(hp):
    return constPo * Delta_fHp(hp)

def Qc_lbfPerFt2_fHpKeas(hp, keas):
    mach = Mach_fHpKeas(hp, keas)
    return Qc_lbfPerFt2_fHpMach(hp, mach)

def QcStdDay_lbfPerFt2_fHpKtas(hp, ktas):
    mach = ktas / SpdSndStdDay_kts_fHp(hp)
    return Qc_lbfPerFt2_fHpMach(hp, mach)
    
def Qc_lbfPerFt2_fHpQ(hp, q):
    mach = Mach_fHpQ(hp, q)
    return Qc_lbfPerFt2_fHpMach(hp, mach)

'''def Delta_fHp(hp):
    if (hp < constTropopause11kmInFt):
        delta = np.power(1 + hp * constLapseTrop / constTo, -constGo / constLapseTrop / constRAir)
    elif (hp < constTopIsoThermLayerStrat20kmInFt): #Isothermal layer of lower stratosphere
        delta = constDeltaTropopause * np.exp(constGo / constRAir / constOatIsoLayerStrat11to20kmInK * (constTropopause11kmInFt - hp))
    elif (hp < constTop1stInverLayerStrat32kmInFt): #First inversion layer of stratosphere
        delta = constDeltaStrat20km * np.power(1 + (hp - constTopIsoThermLayerStrat20kmInFt) * constLapseStrat2 / constOatIsoLayerStrat11to20kmInK, -constGo / constLapseStrat2 / constRAir)
    elif (hp < constStratopauseStart47kmInFt): #Second inversion layer of stratosphere
        delta = constDeltaStrat32km * np.power(1 + (hp - constTop1stInverLayerStrat32kmInFt) * constLapseStrat3 / constOatStrat32kmInK, -constGo / constLapseStrat3 / constRAir)
    elif (hp < constStratopauseEnd51kmInFt): #'Stratopause (isothermal layer 47 to 51 km)
        delta = constDeltaStrat47km * np.exp(constGo / constRAir / constOatStratopause47to51kmInK * (constStratopauseStart47kmInFt - hp))
    else: #Outside bounds of this model - return error code
        delta = constAeroErr
    return delta
'''

def Delta_fHp(hp):
    # Convert hp to a NumPy array if it's not already one
    hp = np.asarray(hp)

    # Initialize the delta array with the same shape as hp
    delta = np.empty(hp.shape)
    
    # Conditions for different altitude ranges
    cond1 = (hp < constTropopause11kmInFt)
    cond2 = (hp >= constTropopause11kmInFt) & (hp < constTopIsoThermLayerStrat20kmInFt)
    cond3 = (hp >= constTopIsoThermLayerStrat20kmInFt) & (hp < constTop1stInverLayerStrat32kmInFt)
    cond4 = (hp >= constTop1stInverLayerStrat32kmInFt) & (hp < constStratopauseStart47kmInFt)
    cond5 = (hp >= constStratopauseStart47kmInFt) & (hp < constStratopauseEnd51kmInFt)
    cond6 = (hp >= constStratopauseEnd51kmInFt)
    
    # Calculate delta for each condition
    delta[cond1] = np.power(1 + hp[cond1] * constLapseTrop / constTo, -constGo / constLapseTrop / constRAir)
    delta[cond2] = constDeltaTropopause * np.exp(constGo / constRAir / constOatIsoLayerStrat11to20kmInK * (constTropopause11kmInFt - hp[cond2]))
    delta[cond3] = constDeltaStrat20km * np.power(1 + (hp[cond3] - constTopIsoThermLayerStrat20kmInFt) * constLapseStrat2 / constOatIsoLayerStrat11to20kmInK, -constGo / constLapseStrat2 / constRAir)
    delta[cond4] = constDeltaStrat32km * np.power(1 + (hp[cond4] - constTop1stInverLayerStrat32kmInFt) * constLapseStrat3 / constOatStrat32kmInK, -constGo / constLapseStrat3 / constRAir)
    delta[cond5] = constDeltaStrat47km * np.exp(constGo / constRAir / constOatStratopause47to51kmInK * (constStratopauseStart47kmInFt - hp[cond5]))
    delta[cond6] = constAeroErr
    
    # If the input was a single float, convert the output to a float
    if delta.size == 1:
        return delta.item()
    return delta

def SpdSnd_ftPerSec_fOatKelvin(oatKelvin):
    return np.sqrt(constGammaAir * constRAir * oatKelvin)
    
def SpdSnd_kts_fOatKelvin(oatKelvin):
    return SpdSnd_ftPerSec_fOatKelvin(oatKelvin) * constConvFtPerSecToKts

def SpdSnd_milePerHr_fOatKelvin(oatKelvin):
    return SpdSnd_ftPerSec_fOatKelvin(oatKelvin) * constConvHrToSec / constConvMileToFt

def SpdSnd_meterPerSec_fOatKelvin(oatKelvin):
    return SpdSnd_ftPerSec_fOatKelvin(oatKelvin) * constConvFtToMeter

def SpdSnd_kmPerHr_fOatKelvin(oatKelvin):
    return SpdSnd_ftPerSec_fOatKelvin(oatKelvin) * constConvHrToSec * constConvFtToMeter / 1000

def SpdSnd_ftPerSec_fOatCelsius(oatCelsius):
    return SpdSnd_ftPerSec_fOatKelvin(ConvCelsiusToKelvin(oatCelsius))

def SpdSnd_kts_fOatCelsius(oatCelsius):
    return SpdSnd_kts_fOatKelvin(ConvCelsiusToKelvin(oatCelsius))

def SpdSnd_milePerHr_fOatCelsius(oatCelsius):
    return SpdSnd_milePerHr_fOatKelvin(ConvCelsiusToKelvin(oatCelsius))

def SpdSnd_meterPerSec_fOatCelsius(oatCelsius):
    return SpdSnd_meterPerSec_fOatKelvin(ConvCelsiusToKelvin(oatCelsius))

def SpdSnd_kmPerHr_fOatCelsius(oatCelsius):
    return SpdSnd_kmPerHr_fOatKelvin(ConvCelsiusToKelvin(oatCelsius))

def SpdSnd_ftPerSec_fOatRankine(oatRankine):
    return SpdSnd_ftPerSec_fOatKelvin(ConvRankineToKelvin(oatRankine))

def SpdSnd_kts_fOatRankine(oatRankine):
    return SpdSnd_kts_fOatKelvin(ConvRankineToKelvin(oatRankine))

def SpdSnd_milePerHr_fOatRankine(oatRankine):
    return SpdSnd_milePerHr_fOatKelvin(ConvRankineToKelvin(oatRankine))

def SpdSnd_meterPerSec_fOatRankine(oatRankine):
    return SpdSnd_meterPerSec_fOatKelvin(ConvRankineToKelvin(oatRankine))

def SpdSnd_kmPerHr_fOatRankine(oatRankine):
    return SpdSnd_kmPerHr_fOatKelvin(ConvRankineToKelvin(oatRankine))

def SpdSnd_ftPerSec_fOatFahrenheit(oatFahrenheit):
    return SpdSnd_ftPerSec_fOatKelvin(ConvFahrenheitToKelvin(oatFahrenheit))

def SpdSnd_kts_fOatFahrenheit(oatFahrenheit):
    return SpdSnd_kts_fOatKelvin(ConvFahrenheitToKelvin(oatFahrenheit))

def SpdSnd_milePerHr_fOatFahrenheit(oatFahrenheit):
    return SpdSnd_milePerHr_fOatKelvin(ConvFahrenheitToKelvin(oatFahrenheit))

def SpdSnd_meterPerSec_fOatFahrenheit(oatFahrenheit):
    return SpdSnd_meterPerSec_fOatKelvin(ConvFahrenheitToKelvin(oatFahrenheit))

def SpdSnd_kmPerHr_fOatFahrenheit(oatFahrenheit):
    return SpdSnd_kmPerHr_fOatKelvin(ConvFahrenheitToKelvin(oatFahrenheit))

def SpdSndStdDay_ftPerSec_fHp(hp):
    return SpdSnd_ftPerSec_fOatKelvin(OatStdDay_Kelvin_fHp(hp))

def SpdSndStdDay_kts_fHp(hp):
    return SpdSndStdDay_ftPerSec_fHp(hp) * constConvFtPerSecToKts

def SpdSndStdDay_milePerHr_fHp(hp):
    return SpdSndStdDay_ftPerSec_fHp(hp) * constConvHrToSec / constConvMileToFt

def SpdSndStdDay_meterPerSec_fHp(hp):
    return SpdSndStdDay_ftPerSec_fHp(hp) * constConvFtToMeter

def SpdSndStdDay_kmPerHr_fHp(hp):
    return SpdSndStdDay_ftPerSec_fHp(hp) * constConvHrToSec * constConvFtToMeter / 1000

def SpdSnd_ftPerSec_fIsaDevCelsiusHp(isaDevCelsius, hp):
    return SpdSnd_ftPerSec_fOatKelvin(Oat_Kelvin_fIsaDevCelsiusHp(isaDevCelsius, hp))

def SpdSnd_kts_fIsaDevCelsiusHp(isaDevCelsius, hp):
    return SpdSnd_kts_fOatKelvin(Oat_Kelvin_fIsaDevCelsiusHp(isaDevCelsius, hp))

def SpdSnd_milePerHr_fIsaDevCelsiusHp(isaDevCelsius, hp):
    return SpdSnd_milePerHr_fOatKelvin(Oat_Kelvin_fIsaDevCelsiusHp(isaDevCelsius, hp))

def SpdSnd_meterPerSec_fIsaDevCelsiusHp(isaDevCelsius, hp):
    return SpdSnd_meterPerSec_fOatKelvin(Oat_Kelvin_fIsaDevCelsiusHp(isaDevCelsius, hp))

def SpdSnd_kmPerHr_fIsaDevCelsiusHp(isaDevCelsius, hp):
    return SpdSnd_kmPerHr_fOatKelvin(Oat_Kelvin_fIsaDevCelsiusHp(isaDevCelsius, hp))

def SpdSnd_ftPerSec_fIsaDevFahrenheitHp(isaDevFahrenheit, hp):
    return SpdSnd_ftPerSec_fOatRankine(Oat_Rankine_fIsaDevFahrenheitHp(isaDevFahrenheit, hp))

def SpdSnd_kts_fIsaDevFahrenheitHp(isaDevFahrenheit, hp):
    return SpdSnd_kts_fOatRankine(Oat_Rankine_fIsaDevFahrenheitHp(isaDevFahrenheit, hp))

def SpdSnd_milePerHr_fIsaDevFahrenheitHp(isaDevFahrenheit, hp):
    return SpdSnd_milePerHr_fOatRankine(Oat_Rankine_fIsaDevFahrenheitHp(isaDevFahrenheit, hp))

def SpdSnd_meterPerSec_fIsaDevFahrenheitHp(isaDevFahrenheit, hp):
    return SpdSnd_meterPerSec_fOatRankine(Oat_Rankine_fIsaDevFahrenheitHp(isaDevFahrenheit, hp))

def SpdSnd_kmPerHr_fIsaDevFahrenheitHp(isaDevFahrenheit, hp):
    return SpdSnd_kmPerHr_fOatRankine(Oat_Rankine_fIsaDevFahrenheitHp(isaDevFahrenheit, hp))

def Keas_fHpKtasOatKelvin(hp, ktas, oatKelvin):
    mach = Mach_fKtasOatKelvin(ktas, oatKelvin)
    return Keas_fHpMach(hp, mach)

def Keas_fHpKtasOatCelsius(hp, ktas, oatCelsius):
    mach = Mach_fKtasOatCelsius(ktas, oatCelsius)
    return Keas_fHpMach(hp, mach)

def Keas_fHpKtasOatRankine(hp, ktas, oatRankine):
    mach = Mach_fKtasOatRankine(ktas, oatRankine)
    return Keas_fHpMach(hp, mach)

def Keas_fHpKtasOatFahrenheit(hp, ktas, oatFahrenheit):
    mach = Mach_fKtasOatFahrenheit(ktas, oatFahrenheit)
    return Keas_fHpMach(hp, mach)

def Keas_fHpKtasIsaDevCelsius(hp, ktas, isaDevCelsius):
    mach = Mach_fHpKtasIsaDevCelsius(hp, ktas, isaDevCelsius)
    return Keas_fHpMach(hp, mach)

def Keas_fHpKtasIsaDevFahrenheit(hp, ktas, isaDevFahrenheit):
    mach = Mach_fHpKtasIsaDevFahrenheit(hp, ktas, isaDevFahrenheit)
    return Keas_fHpMach(hp, mach)

def Mach_fKtasOatKelvin(ktas, oatKelvin):
    return ktas / SpdSnd_kts_fOatKelvin(oatKelvin)

def Mach_fKtasOatCelsius(ktas, oatCelsius):
    return ktas / SpdSnd_kts_fOatCelsius(oatCelsius)

def Mach_fKtasOatRankine(ktas, oatRankine):
    return ktas / SpdSnd_kts_fOatRankine(oatRankine)

def Mach_fKtasOatFahrenheit(ktas, oatFahrenheit):
    return ktas / SpdSnd_kts_fOatFahrenheit(oatFahrenheit)

def Mach_fHpKtasIsaDevCelsius(hp, ktas, isaDevCelsius):
    oatCelsius = OatStdDay_Celsius_fHp(hp) + isaDevCelsius
    return ktas / SpdSnd_kts_fOatCelsius(oatCelsius)

def Mach_fHpKtasIsaDevFahrenheit(hp, ktas, isaDevFahrenheit):
    oatFahrenheit = OatStdDay_Fahrenheit_fHp(hp) + isaDevFahrenheit
    return ktas / SpdSnd_kts_fOatFahrenheit(oatFahrenheit)

def Keas_fHpKcas(hp, kcas):
    mach = Mach_fHpKcas(hp, kcas)
    return Keas_fHpMach(hp, mach)

def KeasStdDay_fHpKtas(hp, ktas):
    return ktas * np.sqrt(SigmaStdDay_fHp(hp))

def Keas_fHpMach(hp, mach):
    return KtasStdDay_fHpMach(hp, mach) * np.sqrt(SigmaStdDay_fHp(hp))

def Keas_fQ(q):
    return constConvFtPerSecToKts * np.sqrt(2 * q / constRhoo)

def Keas_fHpQc(hp, qc):
    mach = Mach_fHpQc(hp, qc)
    return Keas_fHpMach(hp, mach)

def Rho_slugPerFt3_fHpOatKelvin(hp, oatKelvin):
    return constRhoo * Sigma_fOatKelvinHp(oatKelvin, hp)

def Rho_slugPerFt3_fHpOatCelsius(hp, oatCelsius):
    return constRhoo * Sigma_fOatCelsiusHp(oatCelsius, hp)

def Rho_slugPerFt3_fHpOatRankine(hp, oatRankine):
    return constRhoo * Sigma_fOatRankineHp(oatRankine, hp)

def Rho_slugPerFt3_fHpOatFahrenheit(hp, oatFahrenheit):
    return constRhoo * Sigma_fOatFahrenheitHp(oatFahrenheit, hp)

def Rho_slugPerFt3_fHpIsaDevCelsius(hp, isaDevCelsius):
    return constRhoo * Sigma_fIsaDevCelsiusHp(isaDevCelsius, hp)

def Rho_slugPerFt3_fHpIsaDevFahrenheit(hp, isaDevFahrenheit):
    return constRhoo * Sigma_fIsaDevFahrenheitHp(isaDevFahrenheit, hp)

def RhoStdDay_slugPerFt3_fHp(hp):
    return constRhoo * SigmaStdDay_fHp(hp)

def Sigma_fOatKelvinHp(oatKelvin, hp):
    return Delta_fHp(hp) / Theta_fOatKelvin(oatKelvin)

def Sigma_fOatCelsiusHp(oatCelsius, hp):
    return  Delta_fHp(hp) / Theta_fOatCelsius(oatCelsius)

def Sigma_fOatRankineHp(oatRankine, hp):
    return Delta_fHp(hp) / Theta_fOatRankine(oatRankine)

def Sigma_fOatFahrenheitHp(oatFahrenheit, hp):
    return Delta_fHp(hp) / Theta_fOatFahrenheit(oatFahrenheit)

def Theta_fOatKelvin(oatKelvin):
    return  oatKelvin / constTo

def Theta_fOatCelsius(oatCelsius):
    return Theta_fOatKelvin(ConvCelsiusToKelvin(oatCelsius))

def Theta_fOatRankine(oatRankine):
    return Theta_fOatKelvin(ConvRankineToKelvin(oatRankine))

def Theta_fOatFahrenheit(oatFahrenheit):
    return Theta_fOatKelvin(ConvFahrenheitToKelvin(oatFahrenheit))

def Sigma_fIsaDevCelsiusHp(isaDevCelsius, hp):
    return Delta_fHp(hp) / Theta_fIsaDevCelsiusHp(isaDevCelsius, hp)

def Theta_fIsaDevCelsiusHp(isaDevCelsius, hp):
    return Oat_Kelvin_fIsaDevCelsiusHp(isaDevCelsius, hp) / constTo

def Sigma_fIsaDevFahrenheitHp(isaDevFahrenheit, hp):
    return Delta_fHp(hp) / Theta_fIsaDevFahrenheitHp(isaDevFahrenheit, hp)

def Theta_fIsaDevFahrenheitHp(isaDevFahrenheit, hp):
    return Oat_Rankine_fIsaDevFahrenheitHp(isaDevFahrenheit, hp) / constTo / constConvKelvinToRankine

def KtasStdDay_fHpKcas(hp, kcas):
    return SpdSndStdDay_kts_fHp(hp) * Mach_fHpKcas(hp, kcas)

def KtasStdDay_fHpKeas(hp, keas):
    return keas / np.sqrt(SigmaStdDay_fHp(hp))

def KtasStdDay_fHpMach(hp, mach):
    return SpdSndStdDay_kts_fHp(hp) * mach

def KtasStdDay_fHpQ(hp, q):
    return constConvFtPerSecToKts * np.sqrt(2 * q / (RhoStdDay_slugPerFt3_fHp(hp)))

def Ktas_fHpQOatKelvin(hp, q, oatKelvin):
    return Mach_fHpQ(hp, q) * SpdSnd_kts_fOatKelvin(oatKelvin)

def Ktas_fHpQOatCelsius(hp, q, oatCelsius):
    return Mach_fHpQ(hp, q) * SpdSnd_kts_fOatCelsius(oatCelsius)

def Ktas_fHpQOatRankine(hp, q, oatRankine):
    return Mach_fHpQ(hp, q) * SpdSnd_kts_fOatRankine(oatRankine)

def Ktas_fHpQOatFahrenheit(hp, q, oatFahrenheit):
    return Mach_fHpQ(hp, q) * SpdSnd_kts_fOatFahrenheit(oatFahrenheit)

def Ktas_fHpQIsaDevCelsius(hp, q, isaDevCelsius):
    return Mach_fHpQ(hp, q) * SpdSnd_kts_fIsaDevCelsiusHp(isaDevCelsius, hp)

def Ktas_fHpQIsaDevFahrenheit(hp, q, isaDevFahrenheit):
    return Mach_fHpQ(hp, q) * SpdSnd_kts_fIsaDevFahrenheitHp(isaDevFahrenheit, hp)

def KtasStdDay_fHpQc(hp, qc):
    return Mach_fHpQc(hp, qc) * SpdSndStdDay_kts_fHp(hp)

def Ktas_fHpQcOatKelvin(hp, qc, oatKelvin):
    return Mach_fHpQc(hp, qc) * SpdSnd_kts_fOatKelvin(oatKelvin)

def Ktas_fHpQcOatCelsius(hp, qc, oatCelsius):
    return Mach_fHpQc(hp, qc) * SpdSnd_kts_fOatCelsius(oatCelsius)

def Ktas_fHpQcOatRankine(hp, qc, oatRankine):
    return Mach_fHpQc(hp, qc) * SpdSnd_kts_fOatRankine(oatRankine)

def Ktas_fHpQcOatFahrenheit(hp, qc, oatFahrenheit):
    return Mach_fHpQc(hp, qc) * SpdSnd_kts_fOatFahrenheit(oatFahrenheit)

def Ktas_fHpQcIsaDevCelsius(hp, qc, isaDevCelsius):
    return Mach_fHpQc(hp, qc) * SpdSnd_kts_fIsaDevCelsiusHp(isaDevCelsius, hp)

def Ktas_fHpQcIsaDevFahrenheit(hp, qc, isaDevFahrenheit):
    return Mach_fHpQc(hp, qc) * SpdSnd_kts_fIsaDevFahrenheitHp(isaDevFahrenheit, hp)

def Q_lbfPerFt2_fHpKcas(hp, kcas):
    mach = Mach_fHpKcas(hp, kcas)
    return Q_lbfPerFt2_fHpMach(hp, mach)

def Q_lbfPerFt2_fKeas(keas):
    return 0.5 * constRhoo * ConvKtsToFtPerSec(keas)**2

def Q_lbfPerFt2_fHpKtasOatKelvin(hp, ktas, oatKelvin):
    mach = Mach_fKtasOatKelvin(ktas, oatKelvin)
    return Q_lbfPerFt2_fHpMach(hp, mach)

def Q_lbfPerFt2_fHpKtasOatCelsius(hp, ktas, oatCelsius):
    mach = Mach_fKtasOatCelsius(ktas, oatCelsius)
    return Q_lbfPerFt2_fHpMach(hp, mach)

def Q_lbfPerFt2_fHpKtasOatRankine(hp, ktas, oatRankine):
    mach = Mach_fKtasOatRankine(ktas, oatRankine)
    return Q_lbfPerFt2_fHpMach(hp, mach)

def Q_lbfPerFt2_fHpKtasOatFahrenheit(hp, ktas, oatFahrenheit):
    mach = Mach_fKtasOatFahrenheit(ktas, oatFahrenheit)
    return Q_lbfPerFt2_fHpMach(hp, mach)

def Q_lbfPerFt2_fHpKtasIsaDevCelsius(hp, ktas, isaDevCelsius):
    mach = Mach_fHpKtasIsaDevCelsius(hp, ktas, isaDevCelsius)
    return Q_lbfPerFt2_fHpMach(hp, mach)

def Q_lbfPerFt2_fHpKtasIsaDevFahrenheit(hp, ktas, isaDevFahrenheit):
    mach = Mach_fHpKtasIsaDevFahrenheit(hp, ktas, isaDevFahrenheit)
    return Q_lbfPerFt2_fHpMach(hp, mach)

def QStdDay_lbfPerFt2_fHpKtas(hp, ktas):
    mach = ktas / SpdSndStdDay_kts_fHp(hp)
    return Q_lbfPerFt2_fHpMach(hp, mach)

def Q_lbfPerFt2_fHpMach(hp, mach):
    return constGammaAir / 2 * Delta_fHp(hp) * constPo * np.square(mach)

def Q_lbfPerFt2_fHpQc(hp, qc):
    mach = Mach_fHpQc(hp, qc)
    return Q_lbfPerFt2_fHpMach(hp, mach)

def Mach_fHpKcas(hp, kcas):
    # For Mach < 1 only!
    deltaPressureStatic = Delta_fHp(hp)
    return np.power(2 / (constGammaAir - 1) * (np.power(1 / deltaPressureStatic * (np.power(1 + (constGammaAir - 1) / 2 * (kcas / constAo)**2, constGammaAir / (constGammaAir - 1)) - 1) + 1, (constGammaAir - 1) / constGammaAir) - 1), 0.5)

def Mach_fHpQc(hp, qc):
    deltaPressureStatic = Delta_fHp(hp)
    pStatic = deltaPressureStatic * constPo
    mach = np.power(2 / (constGammaAir - 1) * (np.power(qc / pStatic + 1, (constGammaAir - 1) / constGammaAir) - 1), 0.5)
    if mach > 1:
        count = 0 # Counter included as safety mechanism to prevent infinite loop in event of unconverged solution
        while True:
            machLast = mach
            mach = np.power(2 / (constGammaAir + 1) * np.power(qc / pStatic + 1, (constGammaAir - 1) / constGammaAir) * np.power((1 - constGammaAir + 2 * constGammaAir * machLast**2) / (constGammaAir + 1), 1 / constGammaAir), 0.5)
            count = count + 1
            if count > constMaxIter:
                #Set mach and machLast equal to constAeroErr; terminates loop and returns error-flag value
                mach = constAeroErr
                machLast = constAeroErr
                break
            if np.abs(mach - machLast) < constMachEpsilon:
                break
    return mach
    
def Mach_fHpKeas(hp, keas):
    return ConvKtsToFtPerSec(keas) * np.power(constRhoo / constGammaAir / Pstatic_lbfPerFt2_fHp(hp), 0.5)

def Mach_fHpQ(hp, q):
    return np.power(2 * q / (constGammaAir * constPo * Delta_fHp(hp)), 0.5)

def MachStdDay_fHpKtas(hp, ktas):
    return ktas / SpdSndStdDay_kts_fHp(hp)

def OatStdDay_Kelvin_fHp(hp):
    if np.where(hp < constTropopause11kmInFt): #Troposphere
        OatStdDay_Kelvin = constTo + hp * constLapseTrop
    elif (hp < constTopIsoThermLayerStrat20kmInFt): #Isothermal layer of lower stratosphere
        OatStdDay_Kelvin = constOatIsoLayerStrat11to20kmInK
    elif (hp < constTop1stInverLayerStrat32kmInFt): #First inversion layer of stratosphere
        OatStdDay_Kelvin = constOatIsoLayerStrat11to20kmInK + (hp - constTopIsoThermLayerStrat20kmInFt) * constLapseStrat2
    elif (hp < constStratopauseStart47kmInFt): #Second inversion layer of stratosphere
        OatStdDay_Kelvin = constOatStrat32kmInK + (hp - constTop1stInverLayerStrat32kmInFt) * constLapseStrat3
    elif (hp <= constStratopauseEnd51kmInFt): #Stratopause (isothermal layer 47 to 51 km)
        OatStdDay_Kelvin = constOatStratopause47to51kmInK
    else: #Outside bounds of this model - return error code
        OatStdDay_Kelvin = constAeroErr
    return OatStdDay_Kelvin
'''
To efficiently handle multiple conditional operations on a numpy array, 
use numpy.select for vectorized operations. This approach allows the definition 
of multiple conditions and corresponding choices for each condition, ensuring
that the operations are applied correctly across the array. 
Here’s how you can rewrite your function using numpy.select:

# Define your constants
constTropopause11kmInFt = 36089  # Example value
constTopIsoThermLayerStrat20kmInFt = 65617  # Example value
constTop1stInverLayerStrat32kmInFt = 104987  # Example value
constStratopauseStart47kmInFt = 154199  # Example value
constStratopauseEnd51kmInFt = 167322  # Example value

constTo = 288.15  # Example value
constLapseTrop = -0.00649  # Example value
constOatIsoLayerStrat11to20kmInK = 216.65  # Example value
constLapseStrat2 = 0.001  # Example value
constOatStrat32kmInK = 228.65  # Example value
constLapseStrat3 = 0.0028  # Example value
constOatStratopause47to51kmInK = 270.65  # Example value
constAeroErr = -9999  # Example error code

def OatStdDay_Kelvin_fHp(hp):
    # Define conditions
    conditions = [
        hp < constTropopause11kmInFt,
        (hp >= constTropopause11kmInFt) & (hp < constTopIsoThermLayerStrat20kmInFt),
        (hp >= constTopIsoThermLayerStrat20kmInFt) & (hp < constTop1stInverLayerStrat32kmInFt),
        (hp >= constTop1stInverLayerStrat32kmInFt) & (hp < constStratopauseStart47kmInFt),
        (hp >= constStratopauseStart47kmInFt) & (hp <= constStratopauseEnd51kmInFt)
    ]
    
    # Define corresponding operations
    choices = [
        constTo + hp * constLapseTrop,
        constOatIsoLayerStrat11to20kmInK,
        constOatIsoLayerStrat11to20kmInK + (hp - constTopIsoThermLayerStrat20kmInFt) * constLapseStrat2,
        constOatStrat32kmInK + (hp - constTop1stInverLayerStrat32kmInFt) * constLapseStrat3,
        constOatStratopause47to51kmInK
    ]
    
    # Apply conditions and choices
    OatStdDay_Kelvin = np.select(conditions, choices, default=constAeroErr)
    
    return OatStdDay_Kelvin

# Example usage
hp = np.array([10000, 40000, 70000, 110000, 160000, 200000])
OatStdDay_Kelvin = OatStdDay_Kelvin_fHp(hp)
print(OatStdDay_Kelvin)

'''
def OatStdDay_Celsius_fHp(hp):
    return ConvKelvinToCelsius(OatStdDay_Kelvin_fHp(hp))

def OatStdDay_Rankine_fHp(hp):
    return ConvKelvinToRankine(OatStdDay_Kelvin_fHp(hp))

def OatStdDay_Fahrenheit_fHp(hp):
    return ConvKelvinToFahrenheit(OatStdDay_Kelvin_fHp(hp))

def Oat_Kelvin_fIsaDevCelsiusHp(isaDevCelsius, hp):
    return OatStdDay_Kelvin_fHp(hp) + isaDevCelsius

def Oat_Celsius_fIsaDevCelsiusHp(isaDevCelsius, hp):
    return OatStdDay_Celsius_fHp(hp) + isaDevCelsius

def Oat_Rankine_fIsaDevCelsiusHp(isaDevCelsius, hp):
    return OatStdDay_Rankine_fHp(hp) + isaDevCelsius * constConvKelvinToRankine

def Oat_Fahrenheit_fIsaDevCelsiusHp(isaDevCelsius, hp):
    return OatStdDay_Fahrenheit_fHp(hp) + isaDevCelsius * constConvKelvinToRankine

def Oat_Kelvin_fIsaDevFahrenheitHp(isaDevFahrenheit, hp):
    return OatStdDay_Kelvin_fHp(hp) + isaDevFahrenheit / constConvKelvinToRankine

def Oat_Celsius_fIsaDevFahrenheitHp(isaDevFahrenheit, hp):
    return OatStdDay_Celsius_fHp(hp) + isaDevFahrenheit / constConvKelvinToRankine

def Oat_Rankine_fIsaDevFahrenheitHp(isaDevFahrenheit, hp):
    return OatStdDay_Rankine_fHp(hp) + isaDevFahrenheit

def Oat_Fahrenheit_fIsaDevFahrenheitHp(isaDevFahrenheit, hp):
    return OatStdDay_Fahrenheit_fHp(hp) + isaDevFahrenheit

def TapeAlt_ft_fGeoptlAlt(hGeopotlInFt):
    return hGeopotlInFt * (1 + hGeopotlInFt / (constRadiusEarth - hGeopotlInFt))

def TapeAlt_meter_fGeoptlAlt(hGeopotlInMeter):
    return TapeAlt_ft_fGeoptlAlt(hGeopotlInMeter / constConvFtToMeter) * constConvFtToMeter

def GeoptlAlt_ft_fTapeAlt(hTapeAltInFt):
    return hTapeAltInFt * constRadiusEarth / (hTapeAltInFt + constRadiusEarth)

def GeoptlAlt_meter_fTapeAlt(hTapeAltInMeter):
    return GeoptlAlt_ft_fTapeAlt(hTapeAltInMeter / constConvFtToMeter) * constConvFtToMeter

def ThetaStdDay_fHp(hp):
    return OatStdDay_Kelvin_fHp(hp) / constTo

def SigmaStdDay_fHp(hp):
    return Delta_fHp(hp) / ThetaStdDay_fHp(hp)




####### ADDED 7/12/2024 ------ UNTESTED ##############
def Qnh_lbPerFt2_fHpGeoPtlAlt(hp, geoPtlAlt):
    #return constPo * (1 + constLapseTrop / constTo * (hp - geoPtlAlt)) ^ (-constGo / constRAir / constLapseTrop)
    return constPo * np.power(1 + constLapseTrop / constTo * (hp - geoPtlAlt), -constGo / constRAir / constLapseTrop)

def Qnh_lbPerInch2_fHpGeoPtlAlt(hp, geoPtlAlt):
    return Qnh_lbPerFt2_fHpGeoPtlAlt(hp, geoPtlAlt) / constConvLbfPerInch2ToLbfPerFt2

def Qnh_inHg_fHpGeoPtlAlt(hp, geoPtlAlt):
    return ConvLbfPerFt2ToInHg(Qnh_lbPerFt2_fHpGeoPtlAlt(hp, geoPtlAlt))

def Qnh_hPa_fHpGeoPtlAlt(hp, geoPtlAlt):
    return ConvLbfPerFt2ToHPa(Qnh_lbPerFt2_fHpGeoPtlAlt(hp, geoPtlAlt))

def Hp_fQnhLbfPerFt2GeoPtlAlt(qnh, geoPtlAlt):
    return constTo / constLapseTrop * (np.power(qnh / constPo, -constRAir * constLapseTrop / constGo) - 1) + geoPtlAlt

def Hp_fQnhLbfPerInch2GeoPtlAlt(qnh, geoPtlAlt):
    return Hp_fQnhLbfPerFt2GeoPtlAlt(ConvLbfPerInch2ToLbfPerFt2(qnh), geoPtlAlt)

def Hp_fQnhInHgGeoPtlAlt(qnh, geoPtlAlt):
    return Hp_fQnhLbfPerFt2GeoPtlAlt(ConvInHgToLbfPerFt2(qnh), geoPtlAlt)

def Hp_fQnhHPaGeoPtlAlt(qnh, geoPtlAlt):
    return Hp_fQnhLbfPerFt2GeoPtlAlt(ConvHPaToLbfPerFt2(qnh), geoPtlAlt)