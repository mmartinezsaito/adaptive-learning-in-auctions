
from .main import ns, sp, D, B, D2


def confint_fisherinfo(hes, conflev):
    obs_fisherinfo_atext = hes # assuming the hessian of the negloglhf at the min
    covmat = sp.linalg.inv(hes)  # the inverse of the hessian is an estimator of the asymptotic cov matrix
    siglev = 1 - conflev
    q = 1 - siglev/2
    critlev = -sp.stats.norm.isf(q)
    ci_vec = [critlev * sp.sqrt(covmat[i, i]) for i in range(covmat.shape[0])]
    return ci_vec 


#########################
###  Model selection  ###
#########################

# Bayesian Information Criterion
def bic(L, m, n):
   return -2 * L + n * sp.log10(m)

# Akaike Information Criterion corrected for finite sample bias
def caic(L, m, n):
   aic = -2 * L + 2 * n # uncorrected
   return aic + (2 * n * (n + 1)) / (m - n - 1)


# Selecting model

a = 0.285; b = 1.001 # dr naive init0(1,1,0); L = -1935.242998 
a = 0.444; b = 0.965 # dr unilatup init0(1,1,0); L = -1965.695041
a = 0.477; b = 1.003 # dr unilatup init5(1,1,5); L = -2044.705100  

fp = list()
fp.append(['11111_ffx_dr_naive_avf_11',    -4839.56, [0.025, 5.381, 5.396,  1.252,  1], [], []]) # su, minInbound, L-BFGS-B  
fp.append(['11111_ffx_dr_unilatupd_11',    -4864.24, [0.344, 1.143, 5.228,  2.026,  2.390], [], []]) # su, minInbound
fp.append(['1111_ffx_pu_naive_avf_11',     -5565.28, [       0.081, 5.472,  8.836,  5.042], [], []]) # su 
fp.append(['1111_ffx_pu_unilatupd_11',     -5638.25, [       0.089, 9.777, 12.120,  4.999], [], []]) # su
fp.append(['1111_ffx_naive_3p_nudger',     -3094.51, [       0.121, 0.009,  9.450,  6.443], [], []]) # su
fp.append(['1111_ffx_naive_gausnudger',    -2128.26, [       0.228, 0.075,  1.018,  1.134], [], []]) # su
fp.append(['111_ffx_kalman_nudger',        -2049.24, [       2.556, 6.900,  5.844,       ], [], []])
fp.append(['null',                         -6325.64, [                                   ], [], []])
fp.append(['saturated',                        0.00, ns * [0],                               [], []])
fp.append(['11_ffx_dr_gausnudger1',        -1742.26, [0.430, 1.004,                      ], [], []]) #fp.append(['111_ffx_dr_gausnudger2',-1742.26, [0.430, 1.004, 0.766], [], []])
fp.append(['11111_ffx_dr_naive_avf_101',   -5044.43, [0.280, 1.190,     1,    100,      1], [], []]) # su
fp.append(['11111_ffx_dr_unilatupd_101',   -4809.60, [0.319, 1.027, 5.334,  1.755,  4.214], [], []]) # su 

fp.append(['11_ffx_dr_unilatupd_101',      -4820.04, [0.319, 1.027], [], []])  

fp.append(['1111_ffx_pu_naive_avf_101',    -5464.32, [       0.297, 9.994, 31.973,  4.025], [], []]) # su
fp.append(['1111_ffx_pu_unilatupd_101',    -5129.35, [       0.110, 3.963,  5.301,  1.980], [], []]) # su
fp.append(['11331_ffx_dr_unilatupd_11',    -4671.41, [0.029, 2.351, 4.068, 2.791, 4.664, 3.026, 6.930, 2.205, 2.000], [], []]) # su, minInbound 
fp.append(['11331_ffx_dr_naive_avf_11',    -4566.06, [0.041, 2.326, 4.521, 2.466, 4.886, 2.784, 7.169, 2.551, 2.000], [], []]) # su, minInbound
fp.append(['1331_ffx_pu_naive_avf_11',     -5412.98, [       0.144, 4.548, 2.568, 4.904, 2.728, 6.319, 1.823, 2.653], [], []]) # fa 
fp.append(['1331_ffx_pu_unilatupd_11',     -5139.37, [       0.134, 4.848, 2.114, 4.928, 2.117, 5.373, 1.548, 3.194], [], []]) # su
#18
fp.append(['131_ffx_dr_gausnudger1',       -1741.19, [0.447, 1.010, 4.817, 5.177, 6.483                            ], [], []]) # su
fp.append(['131_ffx_kalman_nudger1',       -1882.53, [       4.366, 3.993, 4.782, 7.679, 1.572                     ], [], []])
fp.append(['11331_ffx_dr_naive_avf_101',   -4230.97, [0.125, 2.176, 4.725, 2.089, 4.466, 2.740, 6.595, 1.911, 2.118], [], []]) # su 
fp.append(['11331_ffx_dr_unilatupd_101',   -4377.43, [0.122, 1.353, 4.279, 2.699, 4.769, 2.428, 6.004, 1.567, 2.769], [], []]) # 
fp.append(['1331_ffx_pu_naive_avf_101',    -4518.00, [       0.395, 4.092, 3.596, 4.843, 3.656, 7.217, 3.180, 1.743], [], []]) # 
fp.append(['1331_ffx_pu_unilatupd_101',    -5084.51, [       0.127, 3.494, 4.518, 5.338, 2.433, 7.953, 3.848, 1.211], [], []]) # fa
#24
fp.append(['111_ffx_naive_gausnudger1',    -2231.15, [       0.134, 0.080, 1.886       ], [], []]) # su 
fp.append(['111_ffx_naive_lepkurnudger1',  -2192.85, [       0.060, 0.052, 1.704       ], [], []]) # su 
fp.append(['1111_ffx_naive_gausnudger2',   -2098.37, [       0.156, 0.049, 1.304, 1.463], [], []]) # su 
fp.append(['1111_ffx_naive_lepkurnudger2', -2128.26, [       0.228, 0.075, 1.018, 1.134], [], []]) # su 
fp.append(['11_ffx_dr_gausnudger1',        -1742.26, [0.430,               1.004       ], [], []]) # su
fp.append(['11_ffx_dr_lepkurnudger1',      -1494.84, [0.719,               0.624       ], [], []]) # su
fp.append(['111_ffx_dr_gausnudger2',       -1735.81, [0.426,               0.958, 1.128], [], []]) # su
fp.append(['111_ffx_dr_lepkurnudger2',     -1490.77, [0.719,               0.593, 0.713], [], []]) # su
fp.append(['1111_ffx_dr_lepkurnudger31',   -1511.86, [       0.100, 0.129, 0.609, 0.707], [], []]) # su
fp.append(['1111_ffx_dr_lepkurnudger32d',  -1490.77, [0.719, 4e-05,        0.593, 0.713], [], []]) # su
fp.append(['1111_ffx_dr_lepkurnudger32u',  -1481.06, [0.705,        0.070, 0.594, 0.689], [], []]) # su
fp.append(['1111_ffx_dr_lepkurnudger33',   -1482.77, [0.728, 0.105,        0.592, 0.700], [], []]) # su
fp.append(['11111_ffx_dr_lepkurnudger4',   -1470.01, [0.684, 0.057, 0.109, 0.586, 0.688], [], []]) # su
fp.append(['1111111_ffx_dr_lepkurnudger5', -1446.45, [0.700,-0.017, 0.061, 0.589, 0.628, 0.693, 0.360], [], []]) # su, overfitted?
fp.append(['11111_ffx_dr_lepkurnudger6',   -1463.53, [0.740,               0.618, 0.600, 0.631, 0.389], [], []]) # su, overfitted?
#fp.append(['33311_ffx_dr_lepkurnudger4',       0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [], []]) # su

fp.append(['11_ffx_kalman_gausnudger1',    -1993.77, [       4.110,        2.306], [], []]) # su
fp.append(['11_ffx_kalman_lepkurnudger1',  -1957.40, [       2.588,        2.789], [], []]) # su
fp.append(['111_ffx_kalman_gausnudger2',   -1870.02, [       3.338, 1.533, 3.276], [], []]) # su
fp.append(['111_ffx_kalman_lepkurnudger2', -1724.51, [       1.853, 1.554, 4.193], [], []]) # su 

fp.append(['11_ffx_dr_unilatupd_101',      -4734.98, [0.115, 1.044], [], []])

ns_s = 60
rp = []
"""using +- std"""
rp.append(['null',                     sp.log10(1/len(B)) * 60, [], [], []])
rp.append(['saturated',                                    0.0, 60 * [0], [], []])
rp.append(['rfx_neuroshyuka_dr_avf_naive_2p',   (100.65, 6.66), [(0.0158, 0.0657), (1.4413, 0.7309)], [], []]) # all isconv
rp.append(['rfx_neuroshyuka_dr_avf_naive11_2p', (101.13, 1.00), [(0.0018, 0.0010), (1.4036, 0.1106)], [], []]) # all isconv, with SEM
rp.append(['rfx_neuroshyuka_dr_avf_cntfac_2p',   (96.19, 9.06), [(0.1961, 0.2397), (1.3687, 0.6084)], [], []]) # all isconv
rp.append(['rfx_neuroshyuka_nvlepkurnudger_4p',  (35.90, 9.99), [(0.3114, 0.1872), (0.1092, 0.1081), (0.7848, 0.2743), (0.8188, 0.4504)], [], []]) # 46/47 isconv
rp.append(['rfx_neuroshyuka_drgausnudger1_2p',   (31.79,11.01), [(0.5627, 0.2547), (0.8921, 0.3785)], [], []]) # all isconv
rp.append(['rfx_neuroshyuka_drlepkurnudger7_5p', (28.04,21.03), [(0.5845, 0.3066), (0.6114, 0.3039), (0.7888, 0.3091), (0.5567, 0.3382), (0.3022, 0.1440)], [], []]) #17/47 conv

fp29 = []
fp29.append(['dr_avf_agent101', -2999.91436802, [0.0, 1.04181586388], {'yterushkina': -100.66607462273076, 'erydkina': -101.49590081372197, 'emikhedova': -114.68805925851763, 'opetrova': -97.129668126925694, 'ezaikina': -96.743987955188487, 'kisaeva': -102.70087250867412, 'pkikot': -120.76524406506984, 'opolshina': -97.894599395298215, 'dkulik': -98.264053908377932, 'myurevich': -103.94952237720013, 'jsaakyan': -137.06603688225013, 'kakopyan': -109.09736934881276, 'mtodua': -104.34296213311019, 'mbaranova': -97.344722771121468, 'ndagaev': -99.60549569901346, 'dpozdeeva': -99.89506785453311, 'ozabrodskaya': -97.148954580891427, 'elevchenko': -100.41164048825263, 'akonik': -109.5743000172332, 'ekucherova': -100.39770194471338, 'rvakhrushev': -103.81153271544113, 'eokorokova': -95.55306663456696, 'vbudykin': -98.370229333174237, 'azalyaev': -102.2527032883126, 'ekarelova': -93.749535693665763, 'msoloreva': -97.248714043946663, 'rryzhikh': -105.10949798265615, 'mivanov': -113.39640232649857, 'taleshkovskaya': -101.24045124655704}])
fp29.append(['dr_avf_agent11', -3012.66217231, [0.0, 0.990300180556], {'rvakhrushev': -105.42662239055143, 'emikhedova': -115.79479590630719, 'ozabrodskaya': -97.770226997961956, 'taleshkovskaya': -101.34451052895716, 'mivanov': -115.19739796122938, 'ekarelova': -95.292745164786112, 'mtodua': -104.44930610148727, 'pkikot': -117.49331815383908, 'rryzhikh': -105.59585684612036, 'kakopyan': -109.38771529606426, 'azalyaev': -102.23161326804549, 'vbudykin': -98.441211352953133, 'mbaranova': -97.638839861205412, 'ekucherova': -101.29583708084431, 'opolshina': -99.165604462888226, 'ezaikina': -97.356410025757128, 'msoloreva': -98.143499125296103, 'dpozdeeva': -101.12128316909437, 'akonik': -110.49563736566358, 'opetrova': -97.54438239371612, 'eokorokova': -96.168701203175203, 'ndagaev': -99.852736833361092, 'kisaeva': -102.49613717930646, 'yterushkina': -101.09036874263865, 'elevchenko': -101.27975362788598, 'myurevich': -104.82877363003381, 'jsaakyan': -134.70241551608927, 'erydkina': -102.0989009845718, 'dkulik': -98.957571140283719}])
fp29.append(['dr_avfcntfac_agent101', -2957.47211189, [0.0878493795975, 1.00181644388], {'yterushkina': -108.6058392656219, 'erydkina': -94.751597377625032, 'emikhedova': -111.41664373110736, 'opetrova': -94.992034340003556, 'ezaikina': -97.944773240067789, 'kisaeva': -104.62814238652555, 'pkikot': -131.41477206688677, 'opolshina': -101.90905390996582, 'dkulik': -97.318372705400549, 'myurevich': -100.68153966889794, 'jsaakyan': -139.79082747721495, 'kakopyan': -105.48473942493014, 'mtodua': -103.42833092567977, 'mbaranova': -94.342498423375872, 'ndagaev': -93.002937340631277, 'dpozdeeva': -92.232226434994914, 'ozabrodskaya': -94.547816108480859, 'elevchenko': -95.818715520030437, 'akonik': -112.08236569951683, 'ekucherova': -94.535181445549611, 'rvakhrushev': -100.89259941856373, 'eokorokova': -90.499093861955572, 'vbudykin': -96.619904633681799, 'azalyaev': -98.622180327131943, 'ekarelova': -96.954808769587942, 'msoloreva': -92.847775605357072, 'rryzhikh': -101.07797940156708, 'mivanov': -107.29258178919221, 'taleshkovskaya': -103.73678059162324}])
fp29.append(['naive_lepkurnudger2', -1341.61756973, [0.199676544805, 0.0644017193221, 1.0626618341, 1.16241818677], {'jsaakyan': -77.299801405254769, 'yterushkina': -53.43182380075983, 'ozabrodskaya': -35.700257961266594, 'erydkina': -38.076574209473236, 'emikhedova': -51.221019367839361, 'mivanov': -57.263076052803306, 'opetrova': -35.819717436644005, 'opolshina': -49.690772383795341, 'rryzhikh': -49.190526148996469, 'azalyaev': -41.107212026897393, 'myurevich': -54.765775148322668, 'ekucherova': -35.855935586119408, 'pkikot': -80.426150567945683, 'eokorokova': -35.49886329201901, 'ekarelova': -40.780902636376709, 'ezaikina': -44.045306348061537, 'vbudykin': -35.71529565928703, 'kisaeva': -49.602554644383048, 'msoloreva': -32.962040276216719, 'elevchenko': -36.959703156953189, 'kakopyan': -52.3259773854275, 'ndagaev': -40.635878082951301, 'dkulik': -44.019868410615025, 'akonik': -51.680736941653052, 'mbaranova': -38.490033471797801, 'mtodua': -47.338321086896194, 'dpozdeeva': -41.479781115632505, 'taleshkovskaya': -46.719087897351734, 'rvakhrushev': -43.514577230371678}])
fp29.append(['dr_gausnudger1', -1137.20942315, [0.381258409263, 1.0907059209], {'jsaakyan': -98.20697790528321, 'yterushkina': -45.566665981057213, 'ozabrodskaya': -28.811648622470731, 'erydkina': -30.246019016007551, 'emikhedova': -47.930523515360193, 'mivanov': -43.982027477699823, 'opetrova': -35.249592190937868, 'opolshina': -38.408254652434486, 'rryzhikh': -40.533479701139854, 'azalyaev': -36.29580968692833, 'myurevich': -32.073334776376157, 'ekucherova': -34.556359842614434, 'pkikot': -62.794979085101147, 'eokorokova': -29.336815108302847, 'ekarelova': -29.851314913046689, 'ezaikina': -35.781162480141617, 'vbudykin': -31.646790053706944, 'kisaeva': -50.206260303707005, 'msoloreva': -33.688016620186374, 'elevchenko': -31.594738408127078, 'kakopyan': -32.198839430016925, 'ndagaev': -37.150560049175326, 'dkulik': -35.542230987596604, 'akonik': -33.909677261142939, 'mbaranova': -29.269498798741811, 'mtodua': -45.659568703832761, 'dpozdeeva': -29.673432562993831, 'taleshkovskaya': -40.386275732603806, 'rvakhrushev': -36.658569278717245}])
fp29.append(['dr_lepkurnudger7', -990.245414987, [0.531029005072, 0.702609061939, 0.79252342973, 0.653085584372, 0.387249139955], {'jsaakyan': -79.098095811203549, 'yterushkina': -48.048196134412066, 'ozabrodskaya': -19.522613697381257, 'erydkina': -15.758937947111379, 'emikhedova': -51.337841422008019, 'mivanov': -32.570688862375462, 'opetrova': -32.461566141490721, 'opolshina': -41.818329171129335, 'rryzhikh': -31.179891823831852, 'azalyaev': -35.358572095166373, 'myurevich': -24.999388086240952, 'ekucherova': -30.028800238495382, 'pkikot': -56.301761215328376, 'eokorokova': -16.430269793316779, 'ekarelova': -24.462045606757311, 'ezaikina': -32.052257682183843, 'vbudykin': -25.552413434727189, 'kisaeva': -52.968580355309136, 'msoloreva': -28.19926783438228, 'elevchenko': -20.019345727645803, 'kakopyan': -26.940189409527576, 'ndagaev': -26.208438769358164, 'dkulik': -32.586903689922075, 'akonik': -35.239416262070002, 'mbaranova': -20.967527104500718, 'mtodua': -51.137204101933662, 'dpozdeeva': -20.654568715826752, 'taleshkovskaya': -44.294505655527232, 'rvakhrushev': -34.047798198264204}])
fp29_nlls = [sp.mean(list(fp29[i][3].values())) for i in range(6)], [sp.stats.sem(list(fp29[i][3].values())) for i in range(6)]


# Econoshyuka
efp = list()
# bcd
efp.append(['null_bcd',                           -12988, [                                   ], [], []])
efp.append(['saturated_bcd',                        0.00, ns * [0],                               [], []])
efp.append(['11331_ffx_dr_naive_avf_101_bcd', -10504.042, [0.0859, 2.0491, 5.1933, 2.9124, 7.4359, 3.2625, 4.6970, 2.3930, 2.], [], []])  
efp.append(['11111_ffx_dr_lepkurnudger6_bcd',  -2789.999, [0.6290,                 0.5091, 0.6336, 0.4589, 0.3961], [], []]) 
efp.append(['11_kde_bcd',                              0, [     0], [], []]) 
# b
efp.append(['null_b',                            -4329.3, [                                   ], [], []])
efp.append(['saturated_b',                             0, ns//3 * [0],                               [], []])
efp.append(['11331_ffx_dr_naive_avf_101_b',    -3340.221, [0.0669, 2.3858, 4.1584, 2.7530, 4.6523, 3.1058, 7.4018, 3.1741, 2.], [], []])  
efp.append(['11111_ffx_dr_lepkurnudger32_b',    -885.759, [0.6828,                 0.4485, 0.5958, 0.4670, 0.3742], [], []]) 
efp.append(['11111_ffx_dr_lepkurnudger6_b',     -857.011, [0.6828,                 0.4485, 0.5958, 0.4670, 0.3742], [], []]) 
efp.append(['11111_ffx_dr_lepkurnudger7_b',     -838.335, [0.5978,                 0.4643, 0.6235, 0.4144, 0.3653], [], []]) 
efp.append(['11_kde_b',                        -3818.682, [1.1247], [], []]) 
# c
efp.append(['null_c',                            -4329.3, [                                   ], [], []])
efp.append(['saturated_c',                             0, ns//3 * [0],                               [], []])
efp.append(['11331_ffx_dr_naive_avf_101_c',    -3593.593, [0.0484, 3.6473, 5.6666, 2.1036, 6.5306, 2.2191, 5.7846, 2.1249, 2.], [], []])  
efp.append(['11111_ffx_dr_lepkurnudger6_c',    -1020.044, [0.536,                  0.547,  0.731,  0.487,  0.424 ], [], []]) 
efp.append(['11111_ffx_dr_lepkurnudger7_c',    -1000.643, [0.5227,                 0.5624, 0.7075, 0.4711, 0.4156], [], []]) 
efp.append(['11_kde_c',                        -3943.998, [0.9973], [], []]) 
# d
efp.append(['null_d',                            -4329.3, [                                   ], [], []])
efp.append(['saturated_d',                             0, ns//3 * [0],                               [], []])
efp.append(['11331_ffx_dr_naive_avf_101_d',    -3195.678, [0.0676, 1.9862, 3.5014, 3.5807, 7.8691, 3.9214, 4.8188, 3.9913, 2.], [], []])  
efp.append(['11111_ffx_dr_lepkurnudger6_d',     -888.004, [0.746 ,                 0.531 , 0.534 , 0.421 , 0.391 ], [], []]) 
efp.append(['11111_ffx_dr_lepkurnudger7_d',     -862.197, [0.6506,                 0.5079, 0.6975, 0.3562, 0.3921], [], []]) 
efp.append(['11111_ffx_dr_fb1_lepkurnudger6_d',-1467.064, [0.0   ,                 1.1684, 1.0614, 1.1828, 0.8620], [], []]) 
efp.append(['11_kde_d',                        -3636.860, [1.8579], [], []]) 

erp = []
"""using +- std"""
erp.append(['null',                    -sp.log10(1/len(B)) * 100, [], [], []])
erp.append(['saturated',                                     0.0, 100 * [0], [], []])
erp.append(['rfx_dravfnaive_2p_b',    (205.797, 14.598), [(0.0012, 0.0050), (1.1486, 0.7097)], [], []]) # 18 isconv
erp.append(['rfx_dravfnaive_2p_c',    (207.414, 10.871), [(0.0036, 0.0152), (1.1318, 0.4557)], [], []]) # 18 isconv
erp.append(['rfx_dravfnaive_2p_d',    (196.932, 15.802), [(0.0019, 0.0056), (1.7656, 1.0633)], [], []]) # 18 isconv
erp.append(['rfx_dravfcntfac_2p_b',   (192.304, 27.154), [(0.1390, 0.1553), (1.4082, 1.0703)], [], []]) # 18 isconv
erp.append(['rfx_dravfcntfac_2p_c',   (192.500, 13.604), [(0.3203, 0.2646), (1.5053, 0.4586)], [], []]) # 18 isconv
erp.append(['rfx_dravfcntfac_2p_d',   (196.202, 15.129), [(0.1783, 0.1779), (1.3914, 0.7480)], [], []]) # 18 isconv
erp.append(['rfx_nvlepkurnudger_4p_b', (69.312, 23.352), [(0.1856, 0.1534), (0.0756, 0.0710), (0.7426, 0.3216), (0.8255, 0.3766)], [], []]) # 17 isconv
erp.append(['rfx_nvlepkurnudger_4p_c', (76.462, 22.917), [(0.2056, 0.1860), (0.0771, 0.0786), (0.8405, 0.3515), (0.9862, 0.4670)], [], []]) # 18 isconv
erp.append(['rfx_nvlepkurnudger_4p_d', (67.007, 27.363), [(0.1839, 0.1384), (0.0439, 0.0665), (0.7268, 0.3705), (0.8396, 0.3593)], [], []]) # 16 isconv
erp.append(['rfx_drgausnudger_2p_b',   (49.243, 25.214), [(0.5441, 0.2503), (0.6897, 0.3038)], [], []]) # 18 isconv
erp.append(['rfx_drgausnudger_2p_c',   (50.276, 23.800), [(0.5561, 0.2462), (0.6971, 0.2980)], [], []]) # 18 isconv
erp.append(['rfx_drgausnudger_2p_d',   (49.846, 25.267), [(0.5402, 0.2292), (0.6980, 0.3079)], [], []]) # 18 isconv
erp.append(['rfx_drlepkurnudger_5p_b', (35.514, 29.910), [(0.6305, 0.2725), (0.4510, 0.2312), (0.6029, 0.2617), (0.3892, 0.2394), (0.3417, 0.0779)], [], []]) # 4 isconv
erp.append(['rfx_drlepkurnudger_5p_c', (39.071, 33.359), [(0.5799, 0.2887), (0.4856, 0.2680), (0.6217, 0.2742), (0.4309, 0.2786), (0.3370, 0.0998)], [], []]) # 4 isconv
erp.append(['rfx_drlepkurnudger_5p_d', (36.223, 30.924), [(0.6354, 0.2563), (0.4407, 0.2309), (0.6480, 0.2861), (0.4205, 0.2271), (0.3530, 0.0820)], [], []]) # 3 isconv
erp.append(['rfx_kde_1p_b',            (202.381, 14.512), [(1.8015, 0.9337), (0.8784, 0.2130)], [], []]) # 18 isconv
erp.append(['rfx_kde_1p_c',            (211.803, 15.423), [(1.3723, 0.6502), (0.6816, 0.2691)], [], []]) # 18 isconv
erp.append(['rfx_kde_1p_d',            (190.822, 21.807), [(2.8326, 1.4570), (0.4874, 0.3438)], [], []]) # 18 isconv


efp29 = []
# b
efp29.append(['null_1p', [0], sp.log10(1/len(B)) * 100, {}])
efp29.append(['dravfB101_2p', [0.0, 0.876789031192], -3764.95824217, {'b10_amusharapov': -204.49243856285187, 'b07_areshetarov': -229.41101897081185, 'b18_anikolenko': -231.51016824128686, 'b02_igolubev': -222.08515659882326, 'b12_edesyatnikova': -190.01160664817064, 'b09_lbagaeva': -209.66257433880321, 'b08_dunishkov': -220.55503832331985, 'b16_aakimova': -212.48916644105805, 'b03_nsavchenko': -199.86582883886507, 'b15_mterskova': -193.88227482119916, 'b14_mgrishutina': -216.44370928390737, 'b11_ntsoy': -199.08945342052658, 'b01_lnovikova': -220.23350095578903, 'b06_imalakhov': -213.00627580828623, 'b17_epolonskaya': -205.45075693733017, 'b04_amenschikova': -209.19551424851343, 'b05_mkuzhuget': -194.66496883981367, 'b13_ysudorgina': -192.90879088857852}])
efp29.append(['dravf_B11_2p', [0.0, 0.822609533637], -3775.87758032, {'b08_dunishkov': -220.31847857945269, 'b07_areshetarov': -229.92225803727214, 'b18_anikolenko': -229.8877481815079, 'b13_ysudorgina': -196.40489976969275, 'b05_mkuzhuget': -196.21010596352141, 'b03_nsavchenko': -201.36571060574607, 'b09_lbagaeva': -209.11544661118413, 'b04_amenschikova': -209.5722729084502, 'b10_amusharapov': -205.8478942663458, 'b12_edesyatnikova': -192.242113935649, 'b16_aakimova': -213.09641868794239, 'b01_lnovikova': -223.30499815420174, 'b02_igolubev': -222.23410594020257, 'b17_epolonskaya': -205.01722914555677, 'b11_ntsoy': -200.4915558211182, 'b15_mterskova': -194.48209662376857, 'b06_imalakhov': -211.02456324567163, 'b14_mgrishutina': -215.33968384749471}])
efp29.append(['dravfcntfacB101_2p', [0.0789575317571, 1.0319959327], -3629.25577605, {'b10_amusharapov': -187.09293539429902, 'b07_areshetarov': -226.75716457835375, 'b18_anikolenko': -213.75187641733234, 'b02_igolubev': -214.56385024646096, 'b12_edesyatnikova': -167.83515383133062, 'b09_lbagaeva': -206.41913687225176, 'b08_dunishkov': -235.20655112017386, 'b16_aakimova': -207.61687070106245, 'b03_nsavchenko': -190.59151545438064, 'b15_mterskova': -170.41499120817969, 'b14_mgrishutina': -221.11566277404108, 'b11_ntsoy': -181.74268947156779, 'b01_lnovikova': -213.55198172650159, 'b06_imalakhov': -209.5889748707977, 'b17_epolonskaya': -203.51941257719341, 'b04_amenschikova': -211.81139967759736, 'b05_mkuzhuget': -192.10853724180089, 'b13_ysudorgina': -175.56707188952652}])
efp29.append(['nvlepkurdnudger_4p', [0.0911661611689, 0.0484025681652, 1.02648671153, 1.12557032285], -1634.44043747, {'b08_dunishkov': -128.79060124275068, 'b07_areshetarov': -100.25031311960875, 'b09_lbagaeva': -100.59280911717114, 'b05_mkuzhuget': -83.282162110691118, 'b03_nsavchenko': -74.702822479607761, 'b11_ntsoy': -79.490001416118915, 'b15_mterskova': -61.448592088839654, 'b18_anikolenko': -68.100672595985813, 'b13_ysudorgina': -78.785781529932223, 'b02_igolubev': -107.96818077008777, 'b10_amusharapov': -77.458710071058618, 'b04_amenschikova': -112.47358536970711, 'b17_epolonskaya': -108.12846770862075, 'b16_aakimova': -78.365573030635673, 'b06_imalakhov': -100.38991024140265, 'b01_lnovikova': -107.93456482322291, 'b12_edesyatnikova': -69.551538403871419, 'b14_mgrishutina': -96.72615134666755}])
efp29.append(['drgausnduger_2p', [0.462037930814, 0.771250220555], -1087.41499461, {'b08_dunishkov': -86.991930866757002, 'b07_areshetarov': -77.002514700499091, 'b09_lbagaeva': -48.151229533537546, 'b05_mkuzhuget': -50.731898805055792, 'b03_nsavchenko': -48.608076885248764, 'b11_ntsoy': -46.248046670094055, 'b15_mterskova': -40.567241257833317, 'b18_anikolenko': -42.96553441888981, 'b13_ysudorgina': -38.211705844712583, 'b02_igolubev': -71.697848914785297, 'b10_amusharapov': -42.725614738704991, 'b04_amenschikova': -81.69270705218527, 'b17_epolonskaya': -78.067205734528812, 'b16_aakimova': -78.830451573820284, 'b06_imalakhov': -49.247076120909114, 'b01_lnovikova': -60.747768083849607, 'b12_edesyatnikova': -37.557476167806669, 'b14_mgrishutina': -107.37066724010685}])
efp29.append(['drlepkurdnudger_5p', [0.597750987932, 0.452569058652, 0.59635746851, 0.425901795757, 0.360274565546], -838.293548791, {'b08_dunishkov': -74.579929832899253, 'b07_areshetarov': -63.737113480940032, 'b09_lbagaeva': -32.025063678267557, 'b05_mkuzhuget': -41.409172454564043, 'b03_nsavchenko': -36.577310429393101, 'b11_ntsoy': -34.87151397636152, 'b15_mterskova': -20.110294773677275, 'b18_anikolenko': -22.345392729231843, 'b13_ysudorgina': -12.669676992168059, 'b02_igolubev': -57.537036926465035, 'b10_amusharapov': -23.316002883897141, 'b04_amenschikova': -75.438333424575063, 'b17_epolonskaya': -91.871350215158373, 'b16_aakimova': -81.045495056173536, 'b06_imalakhov': -35.027032136145962, 'b01_lnovikova': -34.978876065708576, 'b12_edesyatnikova': -8.8731713677623123, 'b14_mgrishutina': -91.880782368055108}])
efp29.append(['kde_1p', [1.12473307346], -3818.68261621, {-212.90459388877375, -227.16114077310016, -224.9065338840536, -202.06245138210852, -206.14226273192281, -234.76653173351005, -227.80261985346891, -213.00136984747968, -211.14592507458769, -202.30031689849872, -202.22866473062038, -212.307414606804, -189.5017489291393, -214.03016508411227, -194.88242638600116, -204.8533210730134, -219.68764468580267, -218.99748464597818}])
# c
efp29.append(['null_1p', [0], sp.log10(1/len(B)) * 100, {}])
efp29.append(['dravfB101_2p', [0.0, 0.969151628524], -3772.77917438, {'c06_akartseva': -211.57772510461209, 'c07_idutov': -197.47206757415452, 'c15_emoskovchenko': -221.53910150859383, 'c04_egalkina': -226.08792651874094, 'c08_abachurina': -210.32280086421068, 'c02_apetrova': -219.24738897513541, 'c17_kkosyak': -199.50931471854588, 'c11_dwoodward': -203.3517643915244, 'c14_tlevina': -234.70132740964152, 'c13_lchichkina': -204.35006448036933, 'c03_aindeeva': -214.96483019448408, 'c01_dsosinsky': -220.55730595685679, 'c12_elukyanov': -199.50057049379984, 'c10_vsubbotina': -205.80748342230953, 'c05_ebyvaltzeva': -196.26883447482771, 'c18_vnee': -203.81812682368914, 'c09_dzhukova': -197.18774145527863, 'c16_tkurbasova': -206.51480001709322}])
efp29.append(['dravfB11_2p', [0.0, 0.917901696648], -3796.3261583, {'c13_lchichkina': -204.26650096919039, 'c14_tlevina': -232.97506719477499, 'c04_egalkina': -226.79558786115501, 'c06_akartseva': -214.42680723517927, 'c16_tkurbasova': -205.93884984425065, 'c03_aindeeva': -213.72264251940948, 'c10_vsubbotina': -206.03038928665961, 'c15_emoskovchenko': -222.87870143642365, 'c05_ebyvaltzeva': -198.32710124551096, 'c09_dzhukova': -199.94160684105074, 'c17_kkosyak': -201.55158699985739, 'c12_elukyanov': -200.77624433889218, 'c02_apetrova': -220.52488201298922, 'c11_dwoodward': -206.60249401123085, 'c08_abachurina': -214.78681824631695, 'c07_idutov': -198.77063062337396, 'c01_dsosinsky': -222.18213727146639, 'c18_vnee': -205.82811036442041}])
efp29.append(['dravfcntfacB101_2p', [0.371946763143, 1.15181137863], -3620.72787592, {'c01_dsosinsky': -193.71885769795963, 'c05_ebyvaltzeva': -202.17040396545983, 'c11_dwoodward': -190.17282744931506, 'c17_kkosyak': -183.58458091641683, 'c18_vnee': -199.03518713280437, 'c15_emoskovchenko': -231.1278082110455, 'c16_tkurbasova': -209.74490097761415, 'c04_egalkina': -191.97436460186196, 'c02_apetrova': -184.9159187677331, 'c13_lchichkina': -213.88636229896335, 'c03_aindeeva': -211.97599559464265, 'c14_tlevina': -189.94169477666634, 'c06_akartseva': -222.26221419630241, 'c10_vsubbotina': -203.61954965437405, 'c07_idutov': -196.80883088061378, 'c08_abachurina': -190.062345421928, 'c12_elukyanov': -202.9663650346221, 'c09_dzhukova': -202.75966833801598}])
efp29.append(['nvlepkurdnudger_4p', [0.117063487902, 0.0, 1.02614995722, 1.00784252905], -1607.39129508, {'c05_ebyvaltzeva': -79.758602040148077, 'c07_idutov': -79.629914371544999, 'c11_dwoodward': -80.716404889774452, 'c14_tlevina': -104.6591875546071, 'c01_dsosinsky': -89.326457182693161, 'c09_dzhukova': -84.647263156709471, 'c16_tkurbasova': -92.877689754586427, 'c10_vsubbotina': -89.687015231712721, 'c12_elukyanov': -68.661891534179048, 'c17_kkosyak': -75.333969166622012, 'c06_akartseva': -94.462610975188795, 'c13_lchichkina': -91.404111037169784, 'c08_abachurina': -92.37477879601316, 'c02_apetrova': -94.486364651271089, 'c18_vnee': -75.335851485327467, 'c04_egalkina': -107.95248084744776, 'c03_aindeeva': -90.783328140478901, 'c15_emoskovchenko': -115.29337426029214}])
efp29.append(['drgausnduger_2p', [0.392376592766, 0.813153746119], -1137.04544778, {'c05_ebyvaltzeva': -53.689117795501744, 'c07_idutov': -66.253640406110463, 'c11_dwoodward': -41.682666348636808, 'c14_tlevina': -55.499545024788468, 'c01_dsosinsky': -68.242403111396399, 'c09_dzhukova': -54.828565813021704, 'c16_tkurbasova': -47.209932517185827, 'c10_vsubbotina': -76.310408846892969, 'c12_elukyanov': -46.014154745124991, 'c17_kkosyak': -44.943371382142189, 'c06_akartseva': -89.388061482558058, 'c13_lchichkina': -63.366717654389539, 'c08_abachurina': -47.545546130598424, 'c02_apetrova': -57.808498726900297, 'c18_vnee': -66.652354273769717, 'c04_egalkina': -51.534816463225518, 'c03_aindeeva': -85.924236253911886, 'c15_emoskovchenko': -120.15141080089691}])
efp29.append(['drlepkurdnudger_5p', [0.511516443434, 0.551253721448, 0.719237372188, 0.47008157387, 0.416837237819], -1006.75807064, {'c05_ebyvaltzeva': -44.042042056661955, 'c07_idutov': -66.549367037924426, 'c11_dwoodward': -25.922816093115447, 'c14_tlevina': -41.888808030111058, 'c01_dsosinsky': -62.526086072516804, 'c09_dzhukova': -54.50187036583047, 'c16_tkurbasova': -33.41766634167054, 'c10_vsubbotina': -78.512147532632724, 'c12_elukyanov': -27.697877715952249, 'c17_kkosyak': -27.576984447840019, 'c06_akartseva': -88.578024700484818, 'c13_lchichkina': -59.080865714481178, 'c08_abachurina': -35.082540333851561, 'c02_apetrova': -46.511038615623214, 'c18_vnee': -65.184071316164108, 'c04_egalkina': -39.272775227761812, 'c03_aindeeva': -84.777662367451995, 'c15_emoskovchenko': -125.63542666576713}])
efp29.append(['kde_1p', [0.997394110824], -3943.99856958, {-211.25114129241697, -241.12212801821701, -202.79072225101942, -237.59565398680775, -202.47738701286923, -208.11229683810453, -221.57595406869612, -214.18854081790977, -212.34948496242626, -201.00533938639597, -206.38308253859088, -251.25877363571306, -212.52120601543993, -209.37045092127144, -200.74915615895694, -257.52036656928283, -209.44313881818951, -244.28374628540041}])
# d
efp29.append(['null_1p', [0], sp.log10(1/len(B)) * 100, {}])
efp29.append(['dravfB101_2p',[0.0, 1.27085341956], -3623.97693154, {'d07_rnagumanov': -182.32989354582006, 'd13_amasterov': -238.16521977975813, 'd15_nlashuk': -187.95679715672497, 'd14_dsultanov': -184.0526104102054, 'd11_ngalyaviev': -191.52269953085067, 'd04_gsapozhnikov': -229.38236688025441, 'd09_adrozdova': -198.43472225776546, 'd18_iperesetskaya': -220.76831117849611, 'd15_evolchenkova': -193.21821514497643, 'd05_asokolova': -208.87378941661444, 'd12_opolshina': -196.24568626832638, 'd06_zcherkasova': -194.31738716269359, 'd08_estarikova': -196.79721787483444, 'd03_edesyatnikova': -191.76926247688763, 'd02_dzagranichnova': -220.6340901719887, 'd01_amatykina': -197.90370071063487, 'd17_psivokhin': -204.94218490663135, 'd10_enurullina': -186.66277666201927}])
efp29.append(['dravfB11_2p', [0.0, 1.19471310502], -3650.6197485, {'d14_dsultanov': -186.44797118118302, 'd06_zcherkasova': -193.19615766002778, 'd01_amatykina': -196.57813707931436, 'd15_evolchenkova': -196.72264034027424, 'd03_edesyatnikova': -195.04451235678712, 'd09_adrozdova': -199.33992384000894, 'd17_psivokhin': -206.18028378639568, 'd12_opolshina': -197.58871520173935, 'd10_enurullina': -187.83589306331473, 'd02_dzagranichnova': -224.38522221291041, 'd08_estarikova': -201.33705879785893, 'd05_asokolova': -211.22964555871732, 'd04_gsapozhnikov': -227.87836884807186, 'd18_iperesetskaya': -219.88827551358011, 'd15_nlashuk': -192.03572823590036, 'd07_rnagumanov': -184.68305366854511, 'd11_ngalyaviev': -194.21793182583099, 'd13_amasterov': -236.03022933025937}])
efp29.append(['dravfcntfacB101_2p', [0.0, 1.27085175868], -3623.97693154, {'d14_dsultanov': -184.05263299271007, 'd18_iperesetskaya': -220.76828577720084, 'd09_adrozdova': -198.43472604426836, 'd02_dzagranichnova': -220.63406494610689, 'd10_enurullina': -186.66279583329404, 'd11_ngalyaviev': -191.52271235068554, 'd12_opolshina': -196.24569291568338, 'd08_estarikova': -196.7972238013941, 'd07_rnagumanov': -182.32991837974572, 'd17_psivokhin': -204.9421801885224, 'd01_amatykina': -197.90370519113054, 'd03_edesyatnikova': -191.76927497448904, 'd05_asokolova': -208.87377956028607, 'd13_amasterov': -238.1651716424191, 'd04_gsapozhnikov': -229.38233022123813, 'd15_nlashuk': -187.95681463684249, 'd15_evolchenkova': -193.21822574893966, 'd06_zcherkasova': -194.31739633014732}])
efp29.append(['nvlepkurdnudger_4p', [0.137693345044, 0.0274080805373, 1.01478348791, 1.28281955825], -1660.70723099, {'d02_dzagranichnova': -131.28685998892064, 'd09_adrozdova': -79.875613414268727, 'd13_amasterov': -141.34223045074282, 'd17_psivokhin': -89.27027617170252, 'd14_dsultanov': -77.076338551111732, 'd05_asokolova': -102.87535153735851, 'd04_gsapozhnikov': -124.60068863497675, 'd06_zcherkasova': -80.015965286796288, 'd15_evolchenkova': -80.832961679672962, 'd15_nlashuk': -78.483872051941816, 'd08_estarikova': -86.519546661431406, 'd10_enurullina': -78.168243548537674, 'd11_ngalyaviev': -84.408614637057866, 'd03_edesyatnikova': -92.435127579906137, 'd07_rnagumanov': -82.37984282176339, 'd12_opolshina': -83.194795717567189, 'd01_amatykina': -63.854150872780657, 'd18_iperesetskaya': -104.08675138137869}])
efp29.append(['drgausnduger_2p', [0.471511788177, 0.813246319785], -1137.15199678, {'d02_dzagranichnova': -45.808465821969378, 'd09_adrozdova': -92.093796010728411, 'd13_amasterov': -133.35128517640919, 'd17_psivokhin': -53.984001779947256, 'd14_dsultanov': -47.760656673253365, 'd05_asokolova': -78.018629755825245, 'd04_gsapozhnikov': -76.202630893701652, 'd06_zcherkasova': -58.732397365168609, 'd15_evolchenkova': -53.371059062613156, 'd15_nlashuk': -44.877445822938157, 'd08_estarikova': -42.651882048586813, 'd10_enurullina': -57.803467420566299, 'd11_ngalyaviev': -48.257253630259818, 'd03_edesyatnikova': -42.616425360072036, 'd07_rnagumanov': -46.514440546340893, 'd12_opolshina': -58.544525470557879, 'd01_amatykina': -57.566274688639119, 'd18_iperesetskaya': -98.99735925368725}])
efp29.append(['drlepkurdnudger_5p', [0.650590134587, 0.509084648709, 0.706324024835, 0.354322219162, 0.392808683583], -862.195892669, {'d02_dzagranichnova': -28.5837522096916, 'd09_adrozdova': -95.418537634389338, 'd13_amasterov': -94.134650890764732, 'd17_psivokhin': -48.116852778041036, 'd14_dsultanov': -35.578913573495832, 'd05_asokolova': -68.703317878949193, 'd04_gsapozhnikov': -36.711547588038421, 'd06_zcherkasova': -40.854912981291811, 'd15_evolchenkova': -33.67277553245318, 'd15_nlashuk': -29.165698623420273, 'd08_estarikova': -21.849717159099583, 'd10_enurullina': -51.570385937839916, 'd11_ngalyaviev': -35.351766608416831, 'd03_edesyatnikova': -22.821604723830653, 'd07_rnagumanov': -25.908738974057915, 'd12_opolshina': -60.875832002623675, 'd01_amatykina': -44.58228611037805, 'd18_iperesetskaya': -88.294601462191451}])
efp29.append(['kde_1p', [1.85797230279], -3636.86060466, {-235.25931244341993, -197.3262680280265, -189.6408335127108, -176.6257963241998, -188.56242621025126, -172.72662891252136, -188.45963019463673, -190.9911485252604, -251.40584804317044, -201.86694989520385, -246.40099086442504, -190.80423408845462, -234.26845262110982, -188.99015709151368, -185.49259664417039, -182.02333700363022, -193.04553902595262, -222.97045522693659}]) 

efp29_nlls = [sp.mean(list(efp29[i][3].values())) if (i+1)%8!=0 else sp.mean(list(efp29[i][3])) for i in range(24)], [sp.stats.sem(list(efp29[i][3].values())) if (i+1)%8!=0 else sp.stats.sem(list(efp29[i][3])) for i in range(24)]









### Estimate parameter variances ###
"""
  How to estimate confidence intervals for the loglikelihood function itself?
   Bootstrapping is ill-posed in trial-by-trial lhf encompassing sequential dependencies
    I could bootstrap over subjects
    but at any rate, bootstrapping wuld require reestimating lhf minimum every resample, intractable
  I can report simply 
   the parameter confints
   and the lhf in each model type
"""

calcFishInfo = False
if calcFishInfo:
    agmodname = "11331_ffx_dr_naive_avf_101"
    agmodind = [i[0] for i in fp].index(agmodname)
    hes = nd.Hessian(nll_ffx_dr_avf) # OR nll_ffx_dr_nudger
    fp[agmodind][3] = hes(fp[agmodind][2])


hessian = [] 
fp[[i[0] for i in fp].index("1111_ffx_naive_gausnudger")][3] = \
              [[ 7.32355200e+03,-8.70976342e+03, 6.24565116e-01, 1.24945360e+03], 
               [-8.70976342e+03, 3.09308572e+04,-1.94711931,    -3.03465594e+03], 
               [ 6.24565116e-01,-1.94711931,     9.02228337e+02, 1.97534777e-01], 
               [ 1.24945360e+03,-3.03465594e+03, 1.97534777e-01, 4.51129846e+02]] 
hessian.append([[139.392221, 11.28439487, 6.43782988], [11.28439487, 54.82915617, -9.50453542], [6.43782988, -9.50453542, 12.83268609]]) # kalman_nudger
hessian.append([[1276.5806, -0.0142820868, -6.15243635], [-0.014282087, 2255.35148, -0.008113076], [-6.15243635, -0.008113076, 113.776092]]) # dr_gausnudger
hessian.append([[1.44216046e+06, 3.69372207e+03, -1.01774610e+04, 1.78311082e+04, -4.68403978e+02], 
                [3.69372207e+03, 1.22129391e+01, 2.16306295e+01, 4.66670673e+01, 4.95410931], 
                [-1.01774610e+04, 2.16306295e+01, 2.81694644e+03, -3.10508116e+03, 1.14542692e+02], 
                [1.78311082e+04, 4.66670673e+01, -3.10508116e+03, 5.21061143e+03, -3.29831177], 
                [-4.68403978e+02, 4.95410931, 1.14542692e+02, -3.29831177, 1.58184163e+01]]) # dr_naive_avf

fp[[i[0] for i in fp].index("11331_ffx_dr_naive_avf_101")][3] = \
              [[ 2.82645937e+04, 2.33617030e+03,-2.80682533e+02, 7.46734194e+01, 3.05405461e+02, 2.11369897e+02, 1.46933374e+02, 7.07031413e+01,-1.31158714e+02],
               [ 2.33617030e+03, 3.01898729e+02,-2.44619468e+00, 3.81355255e+01,-6.67774587e+01, 3.15030688e+01, 7.71729939e+01,-1.58549020e+01, 7.18321036e+01],
               [-2.80682533e+02,-2.44619468e+00, 2.13699283e+02, 7.31173701e+01, 1.62144584e+00,-1.55756349e+01, 6.27162502e-10,-1.00168280e-11, 1.99999453e+01],
               [ 7.46734194e+01, 3.81355255e+01, 7.31173701e+01, 1.32567675e+02, 2.53775271e+00,-2.43776934e+01,-6.01017986e-10, 5.22590595e-11, 3.13022576e+01],
               [ 3.05405461e+02,-6.67774587e+01, 1.62144584e+00, 2.53775271e+00, 3.68366788e+02,-1.14574588e+01, 6.66070924e+00,-1.76333992e+00,-9.12696607e+01],
               [ 2.11369897e+02, 3.15030688e+01,-1.55756349e+01,-2.43776934e+01,-1.14574588e+01, 1.20984074e+02,-6.39685781e+01, 1.69397647e+01,-2.10375516e+01],
               [ 1.46933374e+02, 7.71729939e+01, 6.27162502e-10,-6.01017986e-10, 6.66070924e+00,-6.39685781e+01, 5.91908320e+02,-5.00447251e+02, 8.21574278e+01],
               [ 7.07031413e+01,-1.58549020e+01,-1.00168280e-11, 5.22590595e-11,-1.76333992e+00, 1.69397647e+01,-5.00447251e+02, 6.32211058e+02,-2.17501573e+01],
               [-1.31158714e+02, 7.18321036e+01, 1.99999453e+01, 3.13022576e+01,-9.12696607e+01,-2.10375516e+01, 8.21574278e+01,-2.17501573e+01, 1.47499047e+02]] 
fp[[i[0] for i in fp].index("11111_ffx_dr_lepkurnudger6")][3] = \
               [[-8.96929525e+01,-5.68016561e+01,-6.11324767e+00, 3.21541437e+01, 3.67322969e+00],
                [-5.68016561e+01, 1.48024153e+03,-6.87042464e-11, 7.33044983e-11,-1.08521573e-09],
                [-6.11324767e+00,-6.87042464e-11, 5.36304698e+02,-1.02771716e-11,-4.22304034e-10],
                [ 3.21541437e+01, 7.33044983e-11,-1.02771716e-11, 1.20499287e+03,-2.19969256e-11],
                [ 3.67322969e+00,-1.08521573e-09,-4.22304034e-10,-2.19969256e-11, 5.14312929e+03]] 


conflev = 0.95
for f in fp:
    try:
        f[4] = confint_fisherinfo(f[3], conflev)
    except ValueError:
        pass
    f.append(bic(f[1], ns, len(f[2])))
    f.append(caic(f[1], ns, len(f[2])))
for f in efp:
    try:
        f[4] = confint_fisherinfo(f[3], conflev)
    except ValueError:
        pass
    f.append(bic(f[1], ns, len(f[2])))
    f.append(caic(f[1], ns, len(f[2])))

bsl = sorted([(f[5], f[6], f[0]) for f in fp])
bsl2 = sorted([(f[5], f[6], f[0]) for f in efp])

