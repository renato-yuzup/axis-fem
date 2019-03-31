%%%
%%%  JOB TITLE : LINEAR DYNAMIC EXPLICIT SOLVER TEST SERIES -- TEST 11: BOUNDARY CONDITION INACTIVATION (PRESCRIBED VELOCITY)
%%%  JOB ID    : {56663df5-16ee-4c53-a828-ddcfd787ab79}
%%%  SUBMITED ON : 12 March 2014, 00:07:35
%%%
%%% ------
%%%
%%%  THIS IS STEP  0 : <untitled>

results = [
    4.9999999999999955e-005    2.4499999999999986e-005    9.9999999999999911e-001    2.0000000000000018e+004    -2.1862625140000513e+004 ; 
    9.9999999999999815e-005    9.8999999999999872e-005    1.9999999999999962e+000    2.0000000000000018e+004    -4.6959475663555953e+004 ; 
    1.4999999999999969e-004    2.2349999999999961e-004    2.9999999999999938e+000    2.0000000000000018e+004    -7.1624439596607321e+004 ; 
    1.9999999999999955e-004    3.9799999999999921e-004    3.9999999999999907e+000    2.0000000000000018e+004    -9.1097002485686127e+004 ; 
    2.4999999999999957e-004    6.2249999999999860e-004    4.9999999999999911e+000    1.9999999999999574e+004    -6.8973832134595490e+004 ; 
    3.0000000000000079e-004    8.9699999999999860e-004    6.0000000000000151e+000    1.9999999999999574e+004    -4.5692424492333943e+004 ; 
    3.5000000000000200e-004    1.2214999999999999e-003    7.0000000000000391e+000    1.9999999999999574e+004    -2.1125151430922851e+004 ; 
    4.0000000000000322e-004    1.5960000000000024e-003    8.0000000000000639e+000    2.0000000000000462e+004    -2.7526262638401045e+003 ; 
    4.5000000000000443e-004    2.0205000000000067e-003    9.0000000000000888e+000    2.0000000000001350e+004    -2.4066558293978975e+004 ; 
    5.0000000000000565e-004    2.4950000000000115e-003    1.0000000000000000e+001    1.9999999999887663e+004    -4.7831674488152610e+004 ; 
    5.5000000000000686e-004    2.9950000000000128e-003    1.0000000000000000e+001    0.0000000000000000e+000    -5.1518161520940273e+004 ; 
    6.0000000000000808e-004    3.4950000000000142e-003    1.0000000000000000e+001    0.0000000000000000e+000    -4.3418462213969491e+004 ; 
    6.5000000000000929e-004    3.9950000000000116e-003    1.0000000000000000e+001    0.0000000000000000e+000    1.9272520357825026e+003 ; 
    7.0000000000001051e-004    4.4949999999999912e-003    1.0000000000000000e+001    0.0000000000000000e+000    4.6704746459966904e+004 ; 
    7.5000000000001172e-004    4.9949999999999708e-003    1.0000000000000000e+001    0.0000000000000000e+000    5.0630590061443327e+004 ; 
    8.0000000000001294e-004    5.4949999999999504e-003    1.0000000000000322e+001    3.2152058793144533e-007    4.2530903235202000e+004 ; 
    8.5000000000001416e-004    6.0256249999999676e-003    1.1250000000000353e+001    2.5000000000000357e+004    -3.0134937841436265e+004 ; 
    9.0000000000001537e-004    6.6187499999999858e-003    1.2500000000000384e+001    2.5000000000002132e+004    -1.0710098592482446e+005 ; 
    9.5000000000001659e-004    7.2743750000000048e-003    1.3750000000000416e+001    2.5000000000002132e+004    -1.4097169247526367e+005 ; 
    1.0000000000000152e-003    7.9925000000000256e-003    1.5000000000000382e+001    2.5000000000000357e+004    -1.5451310450055523e+005 ; 
    1.0500000000000110e-003    8.7731250000000430e-003    1.6250000000000277e+001    2.4999999999998581e+004    -8.1697150008923229e+004 ; 
    1.1000000000000068e-003    9.6162500000000536e-003    1.7500000000000171e+001    2.4999999999998581e+004    -7.6586637058189172e+003 ; 
    1.1500000000000026e-003    1.0521875000000061e-002    1.8750000000000068e+001    2.4999999999998581e+004    2.5188315604606069e+004 ; 
    1.1999999999999984e-003    1.1490000000000063e-002    1.9999999999999961e+001    2.4999999999995027e+004    3.5987137982212298e+004 ; 
    1.2499999999999942e-003    1.2527797823421059e-002    1.9965895506375567e+001    3.8445959703841581e+004    4.7156385087151311e+003 ; 
    1.2999999999999900e-003    1.3460623526534073e-002    1.8375656083431011e+001    -3.9790503277093754e+004    -4.8805552255695129e+003 ; 
    1.3499999999999858e-003    1.4350251091318794e-002    1.8113201458586975e+001    -3.7005959024954442e+004    -4.5390133781098566e+003 ; 
    1.3999999999999816e-003    1.5269498336530778e-002    1.9449463812955948e+001    4.3499062214425994e+003    5.3354332796339588e+002 ; 
    1.4499999999999773e-003    1.6304951224779301e-002    1.9509377870172781e+001    -2.1338087239527574e+005    -2.6172504643962573e+004 ; 
    1.4999999999999731e-003    1.7221961474634912e-002    1.7370467714277794e+001    -5.1018944499358367e+003    -6.2577940883378324e+002 ; 
    1.5499999999999689e-003    1.8105049388477044e-002    1.7603704664575268e+001    1.2755900586061447e+005    1.5645913505694793e+004 ; 
    1.5999999999999647e-003    1.9027929698542476e-002    1.9661005631610820e+001    1.3477360105618488e+005    1.6530828934808753e+004 ; 
    1.6499999999999605e-003    2.0059321604356587e-002    2.0767101083677360e+001    -3.7970548406994181e+004    -4.6573263262085320e+003 ; 
    1.6999999999999563e-003    2.0980623972493700e-002    1.8111442500156194e+001    7.2047021453203721e+004    8.8370198434402591e+003 ; 
    1.7499999999999521e-003    2.1876058605648518e-002    1.7825740059983264e+001    -4.5370587185593424e+004    -5.5649875758992494e+003 ; 
    1.7999999999999479e-003    2.2804287812531632e-002    1.9079589492850491e+001    -7.3888342286355983e+004    -9.0628694124092617e+003 ; 
    1.8499999999999437e-003    2.3825151076446536e-002    1.9888009029260793e+001    -8.7561688501442783e+004    -1.0739991233572040e+004 ; 
    1.8999999999999395e-003    2.4745630203798042e-002    1.6815496654857981e+001    4.2256148016567240e+004    5.1829820441959246e+003 ; 
    1.9499999999999353e-003    2.5637958047995096e-002    1.7726923809443495e+001    -1.4239351794766257e+004    -1.7465459616510022e+003 ; 
    1.9999999999999411e-003    2.6562198979039869e-002    2.0062008874854669e+001    8.9963331192338286e+004    1.1034567798823693e+004 ; 
    2.0499999999999477e-003    2.7581247267057227e-002    2.1216443855625197e+001    1.9188582064312919e+004    2.3536001495905684e+003 ; 
    2.0999999999999543e-003    2.8513705704851441e-002    1.7391733710942194e+001    -4.4705935545641347e+004    -5.4834638763380317e+003 ; 
    2.1499999999999610e-003    2.9403801151026893e-002    1.8083120785152719e+001    -7.5109966663728745e+004    -9.2127093176034723e+003 ; 
    2.1999999999999676e-003    3.0327206050973124e-002    1.9367184549303389e+001    6.6935920989829989e+004    8.2101112592181162e+003 ; 
    2.2499999999999742e-003    3.1345565834094008e-002    1.9803177263529289e+001    -4.1974181318798976e+004    -5.1483970571540112e+003 ; 
    2.2999999999999809e-003    3.2280690387439998e-002    1.6423445325746084e+001    -4.8019406768151457e+004    -5.8898819398940523e+003 ; 
    2.3499999999999875e-003    3.3160285269383777e-002    1.8517424501677677e+001    4.9296759086700666e+004    6.0465572272050704e+003 ; 
    2.3999999999999942e-003    3.4090088172691810e-002    1.9863985366275077e+001    8.7800377042637614e+004    1.0769267882799109e+004 ; 
    2.4500000000000008e-003    3.5112982409251814e-002    2.0416137203656099e+001    -1.0672575493826708e+005    -1.3090584387422583e+004 ; 
    2.5000000000000074e-003    3.6048992591130459e-002    1.7317023194933153e+001    -7.2967192189792404e+004    -8.9498845655977057e+003 ; 
    2.5500000000000141e-003    3.6919328604285005e-002    1.8934338722427437e+001    4.3923310848471345e+004    5.3874700400994316e+003 ; 
    2.6000000000000207e-003    3.7857143034886409e-002    1.9078401533425456e+001    2.5843811351154571e+004    3.1699058355746101e+003 ; 
    2.6500000000000273e-003    3.8875803064657891e-002    1.9843167617501418e+001    -6.9792007686362413e+004    -8.5604282369745997e+003 ; 
    2.7000000000000340e-003    3.9808387711931327e-002    1.7149122655237726e+001    7.0326333326071872e+003    8.6259666337844976e+002 ; 
    2.7500000000000406e-003    4.0676801908623897e-002    1.8468916491219236e+001    6.4689659750560560e+004    7.9345932052501539e+003 ; 
    2.8000000000000472e-003    4.1627908511019523e-002    1.8829021673039392e+001    -1.4785521804324281e+002    -1.8135371448394867e+001 ; 
    2.8500000000000539e-003    4.2641023660038052e-002    2.0506077127903680e+001    -5.9094273925238689e+004    -7.2482839786821914e+003 ; 
    2.9000000000000605e-003    4.3573231716174152e-002    1.7564694074702516e+001    -4.7483580305959440e+004    -5.8241594577771921e+003 ; 
    2.9500000000000672e-003    4.4441089741824154e-002    1.8371993855418225e+001    1.4142401439710047e+004    1.7346543914766987e+003 ; 
    3.0000000000000738e-003    4.5397248599227544e-002    1.9318052856247611e+001    7.3053709569578932e+003    8.9604964658061328e+002 ; 
    3.0500000000000804e-003    4.6399315427903258e-002    2.0961646108138993e+001    -2.2273757073564746e+004    -2.7320162482346018e+003 ; 
    3.1000000000000871e-003    4.7335011609404572e-002    1.7030561518029135e+001    -6.2033283240739394e+004    -7.6087719366473229e+003 ; 
    3.1500000000000937e-003    4.8203910426934497e-002    1.7765728785720821e+001    3.0645169448714576e+004    3.7588225725581015e+003 ; 
    3.2000000000001003e-003    4.9162488653796292e-002    1.9365652680233232e+001    2.1935180531762802e+004    2.6904877081495611e+003 ; 
    3.2500000000001070e-003    5.0157429547735763e-002    2.0494899464634386e+001    3.6898522636888015e+003    4.5258356301044489e+002 ; 
    3.3000000000001136e-003    5.1101130865397935e-002    1.6699299968267983e+001    -2.7295296378168277e+004    -3.3479395936322953e+003 ; 
    3.3500000000001202e-003    5.1969130546580748e-002    1.8566291579443217e+001    6.0122942775053860e+004    7.3744566761066071e+003 ; 
    3.4000000000001269e-003    5.2929341709252259e-002    2.0438742749989139e+001    -3.7272205932666970e+004    -4.5716702341360069e+003 ; 
    3.4500000000001335e-003    5.3922413661011422e-002    2.0421662204245216e+001    -6.3902135348224328e+004    -7.8379983893886965e+003 ; 
    3.5000000000001402e-003    5.4871371418617566e-002    1.6661311211324517e+001    -8.5872735131276655e+004    -1.0532830491243538e+004 ; 
    3.5500000000001468e-003    5.5729420398184772e-002    1.8416372846844151e+001    4.5435186726491869e+004    5.5729111154610837e+003 ; 
    3.6000000000001534e-003    5.6687160638077036e-002    1.9654463563417863e+001    2.6096498853617035e+004    3.2008995453548341e+003 ; 
    3.6500000000001601e-003    5.7680098747365439e-002    1.9613454637778492e+001    8.2189994764867151e+004    1.0081119247117538e+004 ; 
    3.7000000000001667e-003    5.8633386448716275e-002    1.7318759525924733e+001    -3.6499136166454896e+003    -4.4768483702118044e+002 ; 
    3.7500000000001733e-003    5.9488650152027679e-002    1.9064681523355489e+001    5.7354465915569293e+004    7.0348855953054262e+003 ; 
    3.8000000000001800e-003    6.0457452506193991e-002    1.9929263666549712e+001    -2.9348270882055116e+004    -3.5997498151097880e+003 ; 
    3.8500000000001866e-003    6.1455383887040839e-002    1.9753407204080528e+001    -5.5142309524749071e+004    -6.7635507152761529e+003 ; 
    3.9000000000001932e-003    6.2403323088312666e-002    1.7390398656602624e+001    -1.4858573787038168e+005    -1.8224974294955922e+004 ; 
    3.9500000000001999e-003    6.3247391036136622e-002    1.7969984160724149e+001    8.0728257150161793e+004    9.9018279447637760e+003 ; 
    4.0000000000002065e-003    6.4217909235100679e-002    1.9322118549333585e+001    9.3563269651824681e+004    1.1476122868831801e+004 ; 
    4.0500000000002132e-003    6.5211604601987327e-002    2.0069851134973959e+001    6.5013297633900904e+004    7.9742894250172021e+003 ; 
    4.1000000000002198e-003    6.6156465306862436e-002    1.8237456512895108e+001    -2.6957781013794869e+004    -3.3065412136261061e+003 ; 
    4.1500000000002264e-003    6.7008948688771455e-002    1.8371581246438204e+001    1.2712923336628152e+005    1.5593199209059338e+004 ; 
    4.2000000000002331e-003    6.7993690890104083e-002    1.9937274839710444e+001    -6.3817887079313121e+004    -7.8276648098856576e+003 ; 
    4.2500000000002397e-003    6.8984588749483863e-002    1.9607444852609419e+001    -1.2592348618916204e+005    -1.5445306742230605e+004 ; 
    4.3000000000002463e-003    6.9922722141760207e-002    1.6943534228973462e+001    -7.0337331107300080e+004    -8.6273155807497860e+003 ; 
    4.3500000000002530e-003    7.0774315070529573e-002    1.7162374705633731e+001    1.5950802071724308e+005    1.9564660909427192e+004 ; 
    4.4000000000002596e-003    7.1753753796309439e-002    2.0213747452975724e+001    2.0820612078751677e+004    2.5537788846969602e+003 ; 
    4.4500000000002662e-003    7.2736740167361993e-002    2.0259504390999769e+001    5.7803125956800352e+004    7.0899165682359699e+003 ; 
    4.5000000000002729e-003    7.3680160649261833e-002    1.8179550172736349e+001    2.4461411974015653e+004    3.0003458665303538e+003 ; 
    4.5500000000002795e-003    7.4543656493862034e-002    1.8255813350616435e+001    3.1020820485657139e+004    3.8048985324064415e+003 ; 
    4.6000000000002862e-003    7.5522761547073020e-002    2.0568655848453112e+001    -1.2543740493485602e+005    -1.5385685822403728e+004 ; 
    4.6500000000002928e-003    7.6505050475269704e-002    1.8836264490572692e+001    -1.3487586243141988e+004    -1.6543371934976444e+003 ; 
    4.7000000000002994e-003    7.7448340186897724e-002    1.6650320186952378e+001    -4.0799573324135468e+004    -5.0043240066969975e+003 ; 
    4.7500000000003061e-003    7.8306417099195030e-002    1.7276434259035621e+001    7.7381729509740209e+004    9.4913553038616756e+003 ; 
    4.8000000000003127e-003    7.9275854162588782e-002    2.0723864926811395e+001    1.0104439252756616e+005    1.2393729592478008e+004 ; 
    4.8500000000003193e-003    8.0265673500089979e-002    1.9617223775464993e+001    1.0931843563507419e+005    1.3408592964366606e+004 ; 
    4.9000000000003260e-003    8.1214459919340204e-002    1.8376197360142346e+001    -1.1581558381095392e+005    -1.4205509008887802e+004 ; 
    4.9500000000003326e-003    8.2073896272491578e-002    1.8411220017797838e+001    -3.8846235542116847e+004    -4.7647348551614004e+003 ; 
    5.0000000000003392e-003    8.3042807563133453e-002    2.0663441134629934e+001    -2.5015446502679941e+004    -3.0683016823989869e+003
];
