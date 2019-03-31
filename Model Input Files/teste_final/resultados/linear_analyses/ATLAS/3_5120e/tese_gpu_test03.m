%%%
%%%  JOB TITLE : FINAL TEST: LINEAR TEST 3 -- COMPRESSIVE PULSE LOAD, 5120 ELEMENTS, dt = 1.25e-7
%%%  JOB ID    : {545aa903-55a0-479d-b99f-6c0ab1db9bd7}
%%%  SUBMITED ON : 17 Febuary 2014, 21:00:43
%%%
%%% ------
%%%
%%%  THIS IS STEP  0 : Transient analysis with no hourglass control

gpu_03 = [
    2.0000000000000003e-006    0.0000000000000000e+000    0.0000000000000000e+000 ; 
    4.0000000000000007e-006    0.0000000000000000e+000    0.0000000000000000e+000 ; 
    5.9999999999999951e-006    0.0000000000000000e+000    0.0000000000000000e+000 ; 
    7.9999999999999895e-006    0.0000000000000000e+000    0.0000000000000000e+000 ; 
    9.9999999999999839e-006    0.0000000000000000e+000    0.0000000000000000e+000 ; 
    1.1999999999999978e-005    -6.8927546492047938e-008    -4.9382716049382715e+001 ; 
    1.3999999999999973e-005    -1.4230025410875153e-007    -4.9382716049382715e+001 ; 
    1.5999999999999969e-005    -2.2987652116103173e-007    -4.9382716049382715e+001 ; 
    1.7999999999999963e-005    -3.1590804208759143e-007    -4.9382716049382715e+001 ; 
    1.9999999999999958e-005    -4.0326142095726395e-007    -4.9382716049382715e+001 ; 
    2.1999999999999952e-005    -4.9484298008781324e-007    -4.9382716049382715e+001 ; 
    2.3999999999999946e-005    -5.7078004494391121e-007    -4.9382716049382715e+001 ; 
    2.5999999999999941e-005    -6.4886298525808747e-007    -4.9382716049382715e+001 ; 
    2.7999999999999935e-005    -7.1809146467244147e-007    -4.9382716049382715e+001 ; 
    2.9999999999999930e-005    -7.8750895743360481e-007    -4.9382716049382715e+001 ; 
    3.1999999999999965e-005    -8.6195686596489036e-007    -4.9382716049382715e+001 ; 
    3.4000000000000013e-005    -9.3487237833500916e-007    -4.9382716049382715e+001 ; 
    3.6000000000000062e-005    -1.0293170925018251e-006    -4.9382716049382715e+001 ; 
    3.8000000000000111e-005    -1.1258674302326428e-006    -4.9382716049382715e+001 ; 
    4.0000000000000159e-005    -1.2125480072664054e-006    -4.9382716049382715e+001 ; 
    4.2000000000000208e-005    -1.3118630464905702e-006    -4.9382716049382715e+001 ; 
    4.4000000000000256e-005    -1.3938481144588275e-006    -4.9382716049382715e+001 ; 
    4.6000000000000305e-005    -1.4631096646556102e-006    -4.9382716049382715e+001 ; 
    4.8000000000000354e-005    -1.5357096776226088e-006    -4.9382716049382715e+001 ; 
    5.0000000000000402e-005    -1.5910442492579806e-006    -4.9382716049382715e+001 ; 
    5.2000000000000451e-005    -1.6605541383144977e-006    -4.9382716049382715e+001 ; 
    5.4000000000000499e-005    -1.7441036540496180e-006    -4.9382716049382715e+001 ; 
    5.6000000000000548e-005    -1.8237812653117078e-006    -4.9382716049382715e+001 ; 
    5.8000000000000597e-005    -1.9197370958362943e-006    -4.9382716049382715e+001 ; 
    6.0000000000000645e-005    -2.0233526243362222e-006    -4.9382716049382715e+001 ; 
    6.2000000000000694e-005    -2.1152807774151570e-006    -4.9382716049382715e+001 ; 
    6.4000000000000742e-005    -2.2026750895107603e-006    -4.9382716049382715e+001 ; 
    6.6000000000000791e-005    -2.2807231564657438e-006    -4.9382716049382715e+001 ; 
    6.8000000000000840e-005    -2.3520318399539044e-006    -4.9382716049382715e+001 ; 
    7.0000000000000888e-005    -2.4127381888701009e-006    -4.9382716049382715e+001 ; 
    7.2000000000000937e-005    -2.4749668962508499e-006    -4.9382716049382715e+001 ; 
    7.4000000000000986e-005    -2.5450349593964005e-006    -4.9382716049382715e+001 ; 
    7.6000000000001034e-005    -2.6055503598724318e-006    0.0000000000000000e+000 ; 
    7.8000000000001083e-005    -2.6164130748340689e-006    0.0000000000000000e+000 ; 
    8.0000000000001131e-005    -2.6222358593507032e-006    0.0000000000000000e+000 ; 
    8.2000000000001180e-005    -2.6473354959861282e-006    0.0000000000000000e+000 ; 
    8.4000000000001229e-005    -2.6524191634483002e-006    0.0000000000000000e+000 ; 
    8.6000000000001277e-005    -2.6413968978274871e-006    0.0000000000000000e+000 ; 
    8.8000000000001326e-005    -2.6325123147775371e-006    0.0000000000000000e+000 ; 
    9.0000000000001374e-005    -2.6189494113508718e-006    0.0000000000000000e+000 ; 
    9.2000000000001423e-005    -2.6102789693544037e-006    0.0000000000000000e+000 ; 
    9.4000000000001472e-005    -2.6099570608052434e-006    0.0000000000000000e+000 ; 
    9.6000000000001520e-005    -2.6094172613200783e-006    0.0000000000000000e+000 ; 
    9.8000000000001569e-005    -2.6239610356552202e-006    0.0000000000000000e+000 ; 
    1.0000000000000162e-004    -2.6327027634232791e-006    0.0000000000000000e+000 ; 
    1.0200000000000167e-004    -2.6303758678767571e-006    0.0000000000000000e+000 ; 
    1.0400000000000171e-004    -2.6333488116897609e-006    0.0000000000000000e+000 ; 
    1.0600000000000176e-004    -2.6318758651026309e-006    0.0000000000000000e+000 ; 
    1.0800000000000181e-004    -2.6265158909194721e-006    0.0000000000000000e+000 ; 
    1.1000000000000186e-004    -2.6207558408215282e-006    0.0000000000000000e+000 ; 
    1.1200000000000191e-004    -2.6170092373229648e-006    0.0000000000000000e+000 ; 
    1.1400000000000196e-004    -2.6212636831897251e-006    0.0000000000000000e+000 ; 
    1.1600000000000201e-004    -2.6237562910300923e-006    0.0000000000000000e+000 ; 
    1.1800000000000206e-004    -2.6255093962374270e-006    0.0000000000000000e+000 ; 
    1.2000000000000210e-004    -2.6256907842719960e-006    0.0000000000000000e+000 ; 
    1.2200000000000215e-004    -2.6264894100696097e-006    0.0000000000000000e+000 ; 
    1.2400000000000198e-004    -2.6318106793002506e-006    0.0000000000000000e+000 ; 
    1.2600000000000182e-004    -2.6207903901612582e-006    0.0000000000000000e+000 ; 
    1.2800000000000165e-004    -2.6220472828588366e-006    0.0000000000000000e+000 ; 
    1.3000000000000148e-004    -2.6268130419550646e-006    0.0000000000000000e+000 ; 
    1.3200000000000131e-004    -2.6182908565393731e-006    0.0000000000000000e+000 ; 
    1.3400000000000114e-004    -2.6266598498084257e-006    0.0000000000000000e+000 ; 
    1.3600000000000097e-004    -2.6255305241227087e-006    0.0000000000000000e+000 ; 
    1.3800000000000081e-004    -2.6262260670427008e-006    0.0000000000000000e+000 ; 
    1.4000000000000064e-004    -2.6328709643948109e-006    0.0000000000000000e+000 ; 
    1.4200000000000047e-004    -2.6203137548415987e-006    0.0000000000000000e+000 ; 
    1.4400000000000030e-004    -2.6229950747612250e-006    0.0000000000000000e+000 ; 
    1.4600000000000013e-004    -2.6260230875684691e-006    0.0000000000000000e+000 ; 
    1.4799999999999997e-004    -2.6199693185180820e-006    0.0000000000000000e+000 ; 
    1.4999999999999980e-004    -2.6226957675899325e-006    0.0000000000000000e+000 ; 
    1.5199999999999963e-004    -2.6217492667587831e-006    0.0000000000000000e+000 ; 
    1.5399999999999946e-004    -2.6286330193829133e-006    0.0000000000000000e+000 ; 
    1.5599999999999929e-004    -2.6287979888083155e-006    0.0000000000000000e+000 ; 
    1.5799999999999912e-004    -2.6248277683192931e-006    0.0000000000000000e+000 ; 
    1.5999999999999896e-004    -2.6293573289506666e-006    0.0000000000000000e+000 ; 
    1.6199999999999879e-004    -2.6227487620723840e-006    0.0000000000000000e+000 ; 
    1.6399999999999862e-004    -2.6258449964223184e-006    0.0000000000000000e+000 ; 
    1.6599999999999845e-004    -2.6214432911567426e-006    0.0000000000000000e+000 ; 
    1.6799999999999828e-004    -2.6177161122825697e-006    0.0000000000000000e+000 ; 
    1.6999999999999811e-004    -2.6301429833171145e-006    0.0000000000000000e+000 ; 
    1.7199999999999795e-004    -2.6198184863567282e-006    0.0000000000000000e+000 ; 
    1.7399999999999778e-004    -2.6240255069854337e-006    0.0000000000000000e+000 ; 
    1.7599999999999761e-004    -2.6304381735761020e-006    0.0000000000000000e+000 ; 
    1.7799999999999744e-004    -2.6248119117040182e-006    0.0000000000000000e+000 ; 
    1.7999999999999727e-004    -2.6291672575246538e-006    0.0000000000000000e+000 ; 
    1.8199999999999711e-004    -2.6235777039697598e-006    0.0000000000000000e+000 ; 
    1.8399999999999694e-004    -2.6211798527975431e-006    0.0000000000000000e+000 ; 
    1.8599999999999677e-004    -2.6273156804946964e-006    0.0000000000000000e+000 ; 
    1.8799999999999660e-004    -2.6207798893163361e-006    0.0000000000000000e+000 ; 
    1.8999999999999643e-004    -2.6207119269715510e-006    0.0000000000000000e+000 ; 
    1.9199999999999626e-004    -2.6196736347839577e-006    0.0000000000000000e+000 ; 
    1.9399999999999610e-004    -2.6156677963182924e-006    0.0000000000000000e+000 ; 
    1.9599999999999593e-004    -2.6121000757343660e-006    0.0000000000000000e+000 ; 
    1.9799999999999576e-004    -2.5969850503400935e-006    0.0000000000000000e+000 ; 
    1.9999999999999559e-004    -2.5904827804725726e-006    0.0000000000000000e+000 ; 
    2.0199999999999542e-004    -2.5690555896911575e-006    0.0000000000000000e+000 ; 
    2.0399999999999526e-004    -2.5379465369140507e-006    0.0000000000000000e+000 ; 
    2.0599999999999509e-004    -2.5013842089170853e-006    0.0000000000000000e+000 ; 
    2.0799999999999492e-004    -2.4430613747841113e-006    0.0000000000000000e+000 ; 
    2.0999999999999475e-004    -2.3799786353282763e-006    0.0000000000000000e+000 ; 
    2.1199999999999458e-004    -2.2909389102447352e-006    0.0000000000000000e+000 ; 
    2.1399999999999441e-004    -2.1839779331489711e-006    0.0000000000000000e+000 ; 
    2.1599999999999425e-004    -2.0740751190256230e-006    0.0000000000000000e+000 ; 
    2.1799999999999408e-004    -1.9316550284443154e-006    0.0000000000000000e+000 ; 
    2.1999999999999391e-004    -1.7738457263750720e-006    0.0000000000000000e+000 ; 
    2.2199999999999374e-004    -1.6055930783825599e-006    0.0000000000000000e+000 ; 
    2.2399999999999357e-004    -1.4139329883413201e-006    0.0000000000000000e+000 ; 
    2.2599999999999340e-004    -1.2246744369241556e-006    0.0000000000000000e+000 ; 
    2.2799999999999324e-004    -1.0234845067011243e-006    0.0000000000000000e+000 ; 
    2.2999999999999307e-004    -8.2129988229112858e-007    0.0000000000000000e+000 ; 
    2.3199999999999290e-004    -6.3430039417987375e-007    0.0000000000000000e+000 ; 
    2.3399999999999273e-004    -4.4389271481794529e-007    0.0000000000000000e+000 ; 
    2.3599999999999256e-004    -2.7065758398390779e-007    0.0000000000000000e+000 ; 
    2.3799999999999240e-004    -1.0803116981038226e-007    0.0000000000000000e+000 ; 
    2.3999999999999223e-004    4.0546935317201644e-008    0.0000000000000000e+000 ; 
    2.4199999999999206e-004    1.8088382373738079e-007    0.0000000000000000e+000 ; 
    2.4399999999999189e-004    3.2493819496130988e-007    0.0000000000000000e+000 ; 
    2.4599999999999210e-004    4.5938076845555026e-007    0.0000000000000000e+000 ; 
    2.4799999999999237e-004    6.0295065026615047e-007    0.0000000000000000e+000 ; 
    2.4999999999999263e-004    7.5601426349961678e-007    0.0000000000000000e+000 ; 
    2.5199999999999290e-004    9.0595146046438135e-007    0.0000000000000000e+000 ; 
    2.5399999999999316e-004    1.0696199731794685e-006    0.0000000000000000e+000 ; 
    2.5599999999999343e-004    1.2329562525735929e-006    0.0000000000000000e+000 ; 
    2.5799999999999369e-004    1.3998637423023528e-006    0.0000000000000000e+000 ; 
    2.5999999999999396e-004    1.5822333614719327e-006    0.0000000000000000e+000 ; 
    2.6199999999999423e-004    1.7511788695724204e-006    0.0000000000000000e+000 ; 
    2.6399999999999449e-004    1.9165937846131870e-006    0.0000000000000000e+000 ; 
    2.6599999999999476e-004    2.0718166601130235e-006    0.0000000000000000e+000 ; 
    2.6799999999999502e-004    2.2004539584374044e-006    0.0000000000000000e+000 ; 
    2.6999999999999529e-004    2.3176165581182659e-006    0.0000000000000000e+000 ; 
    2.7199999999999555e-004    2.4250622646032816e-006    0.0000000000000000e+000 ; 
    2.7399999999999582e-004    2.5208696777625451e-006    0.0000000000000000e+000 ; 
    2.7599999999999608e-004    2.6005173968002783e-006    0.0000000000000000e+000 ; 
    2.7799999999999635e-004    2.6627309789121400e-006    0.0000000000000000e+000 ; 
    2.7999999999999661e-004    2.7035141322258351e-006    0.0000000000000000e+000 ; 
    2.8199999999999688e-004    2.7226261821629262e-006    0.0000000000000000e+000 ; 
    2.8399999999999715e-004    2.7370508778528724e-006    0.0000000000000000e+000 ; 
    2.8599999999999741e-004    2.7458022261690494e-006    0.0000000000000000e+000 ; 
    2.8799999999999768e-004    2.7431914880666377e-006    0.0000000000000000e+000 ; 
    2.8999999999999794e-004    2.7371563528794649e-006    0.0000000000000000e+000 ; 
    2.9199999999999821e-004    2.7114121388617152e-006    0.0000000000000000e+000 ; 
    2.9399999999999847e-004    2.6697755925068885e-006    0.0000000000000000e+000 ; 
    2.9599999999999874e-004    2.6338440118965532e-006    0.0000000000000000e+000 ; 
    2.9799999999999900e-004    2.5919871283415171e-006    0.0000000000000000e+000 ; 
    2.9999999999999927e-004    2.5683653184535214e-006    0.0000000000000000e+000 ; 
    3.0199999999999953e-004    2.5600766338423790e-006    0.0000000000000000e+000 ; 
    3.0399999999999980e-004    2.5534565085663516e-006    0.0000000000000000e+000 ; 
    3.0600000000000007e-004    2.5646923442177310e-006    0.0000000000000000e+000 ; 
    3.0800000000000033e-004    2.5753823958974794e-006    0.0000000000000000e+000 ; 
    3.1000000000000060e-004    2.5998113899403199e-006    0.0000000000000000e+000 ; 
    3.1200000000000086e-004    2.6290412613548837e-006    0.0000000000000000e+000 ; 
    3.1400000000000113e-004    2.6539185894238876e-006    0.0000000000000000e+000 ; 
    3.1600000000000139e-004    2.6830693246291658e-006    0.0000000000000000e+000 ; 
    3.1800000000000166e-004    2.6896749647404865e-006    0.0000000000000000e+000 ; 
    3.2000000000000192e-004    2.6919378715070317e-006    0.0000000000000000e+000 ; 
    3.2200000000000219e-004    2.6813310306264111e-006    0.0000000000000000e+000 ; 
    3.2400000000000245e-004    2.6523815269775602e-006    0.0000000000000000e+000 ; 
    3.2600000000000272e-004    2.6255502297512404e-006    0.0000000000000000e+000 ; 
    3.2800000000000299e-004    2.5906943364078072e-006    0.0000000000000000e+000 ; 
    3.3000000000000325e-004    2.5708940488002651e-006    0.0000000000000000e+000 ; 
    3.3200000000000352e-004    2.5689930616853585e-006    0.0000000000000000e+000 ; 
    3.3400000000000378e-004    2.5816327658150911e-006    0.0000000000000000e+000 ; 
    3.3600000000000405e-004    2.6094012537945929e-006    0.0000000000000000e+000 ; 
    3.3800000000000431e-004    2.6194111041810227e-006    0.0000000000000000e+000 ; 
    3.4000000000000458e-004    2.6328429156742479e-006    0.0000000000000000e+000 ; 
    3.4200000000000484e-004    2.6426577535519119e-006    0.0000000000000000e+000 ; 
    3.4400000000000511e-004    2.6443293266494131e-006    0.0000000000000000e+000 ; 
    3.4600000000000537e-004    2.6607299188359691e-006    0.0000000000000000e+000 ; 
    3.4800000000000564e-004    2.6600306404954958e-006    0.0000000000000000e+000 ; 
    3.5000000000000591e-004    2.6469573459488628e-006    0.0000000000000000e+000 ; 
    3.5200000000000617e-004    2.6222090943248374e-006    0.0000000000000000e+000 ; 
    3.5400000000000644e-004    2.5863110681113112e-006    0.0000000000000000e+000 ; 
    3.5600000000000670e-004    2.5740165310078865e-006    0.0000000000000000e+000 ; 
    3.5800000000000697e-004    2.5779684210978046e-006    0.0000000000000000e+000 ; 
    3.6000000000000723e-004    2.5875433754205134e-006    0.0000000000000000e+000 ; 
    3.6200000000000750e-004    2.6043651593752398e-006    0.0000000000000000e+000 ; 
    3.6400000000000776e-004    2.6167984663048825e-006    0.0000000000000000e+000 ; 
    3.6600000000000803e-004    2.6246828704497855e-006    0.0000000000000000e+000 ; 
    3.6800000000000829e-004    2.6299861701565112e-006    0.0000000000000000e+000 ; 
    3.7000000000000856e-004    2.6320593855235690e-006    0.0000000000000000e+000 ; 
    3.7200000000000883e-004    2.6403435879906772e-006    0.0000000000000000e+000 ; 
    3.7400000000000909e-004    2.6272052188871964e-006    0.0000000000000000e+000 ; 
    3.7600000000000936e-004    2.6165960016822608e-006    0.0000000000000000e+000 ; 
    3.7800000000000962e-004    2.6063139596911593e-006    0.0000000000000000e+000 ; 
    3.8000000000000989e-004    2.5909220337924957e-006    0.0000000000000000e+000 ; 
    3.8200000000001015e-004    2.5926505329206760e-006    0.0000000000000000e+000 ; 
    3.8400000000001042e-004    2.5919001152195706e-006    0.0000000000000000e+000 ; 
    3.8600000000001068e-004    2.5893848898038172e-006    0.0000000000000000e+000 ; 
    3.8800000000001095e-004    2.5970013037335865e-006    0.0000000000000000e+000 ; 
    3.9000000000001121e-004    2.5934822589348110e-006    0.0000000000000000e+000 ; 
    3.9200000000001148e-004    2.5981985100489719e-006    0.0000000000000000e+000 ; 
    3.9400000000001175e-004    2.5982811682307602e-006    0.0000000000000000e+000 ; 
    3.9600000000001201e-004    2.5775494173592156e-006    0.0000000000000000e+000 ; 
    3.9800000000001228e-004    2.5533144630966454e-006    0.0000000000000000e+000 ; 
    4.0000000000001254e-004    2.5138215915385292e-006    0.0000000000000000e+000 ; 
    4.0200000000001281e-004    2.4727948471165262e-006    0.0000000000000000e+000 ; 
    4.0400000000001307e-004    2.4263880668914965e-006    0.0000000000000000e+000 ; 
    4.0600000000001334e-004    2.3699069034946661e-006    0.0000000000000000e+000 ; 
    4.0800000000001360e-004    2.3092100318993168e-006    0.0000000000000000e+000 ; 
    4.1000000000001387e-004    2.2276409670972128e-006    0.0000000000000000e+000 ; 
    4.1200000000001413e-004    2.1358141383461394e-006    0.0000000000000000e+000 ; 
    4.1400000000001440e-004    2.0310101967966851e-006    0.0000000000000000e+000 ; 
    4.1600000000001467e-004    1.8992603885740063e-006    0.0000000000000000e+000 ; 
    4.1800000000001493e-004    1.7750151720761958e-006    0.0000000000000000e+000 ; 
    4.2000000000001520e-004    1.6159589694285646e-006    0.0000000000000000e+000 ; 
    4.2200000000001546e-004    1.4427415271252869e-006    0.0000000000000000e+000 ; 
    4.2400000000001573e-004    1.2541075928017910e-006    0.0000000000000000e+000 ; 
    4.2600000000001599e-004    1.0571917165436523e-006    0.0000000000000000e+000 ; 
    4.2800000000001626e-004    8.6160342968460865e-007    0.0000000000000000e+000 ; 
    4.3000000000001652e-004    6.6232564598640116e-007    0.0000000000000000e+000 ; 
    4.3200000000001679e-004    4.5070848688745077e-007    0.0000000000000000e+000 ; 
    4.3400000000001705e-004    2.5969585566430039e-007    0.0000000000000000e+000 ; 
    4.3600000000001732e-004    7.3969648777946585e-008    0.0000000000000000e+000 ; 
    4.3800000000001759e-004    -9.7893432824558682e-008    0.0000000000000000e+000 ; 
    4.4000000000001785e-004    -2.6797954857006985e-007    0.0000000000000000e+000 ; 
    4.4200000000001812e-004    -4.3551138575907562e-007    0.0000000000000000e+000 ; 
    4.4400000000001838e-004    -5.7583566586983195e-007    0.0000000000000000e+000 ; 
    4.4600000000001865e-004    -7.2241798600065008e-007    0.0000000000000000e+000 ; 
    4.4800000000001891e-004    -8.5998180050654166e-007    0.0000000000000000e+000 ; 
    4.5000000000001918e-004    -9.9590419311449718e-007    0.0000000000000000e+000 ; 
    4.5200000000001944e-004    -1.1349642677477719e-006    0.0000000000000000e+000 ; 
    4.5400000000001971e-004    -1.2619574211111925e-006    0.0000000000000000e+000 ; 
    4.5600000000001997e-004    -1.4067137401004315e-006    0.0000000000000000e+000 ; 
    4.5800000000002024e-004    -1.5545853591515014e-006    0.0000000000000000e+000 ; 
    4.6000000000002051e-004    -1.7056970582660380e-006    0.0000000000000000e+000 ; 
    4.6200000000002077e-004    -1.8511099409965817e-006    0.0000000000000000e+000 ; 
    4.6400000000002104e-004    -1.9941336297209525e-006    0.0000000000000000e+000 ; 
    4.6600000000002130e-004    -2.1405274316267762e-006    0.0000000000000000e+000 ; 
    4.6800000000002157e-004    -2.2932645074640671e-006    0.0000000000000000e+000 ; 
    4.7000000000002183e-004    -2.4129281977694472e-006    0.0000000000000000e+000 ; 
    4.7200000000002210e-004    -2.5380142061955631e-006    0.0000000000000000e+000 ; 
    4.7400000000002236e-004    -2.6262888279202564e-006    0.0000000000000000e+000 ; 
    4.7600000000002263e-004    -2.7112303079138845e-006    0.0000000000000000e+000 ; 
    4.7800000000002289e-004    -2.7700595714118125e-006    0.0000000000000000e+000 ; 
    4.8000000000002316e-004    -2.7951169362902953e-006    0.0000000000000000e+000 ; 
    4.8200000000002343e-004    -2.8089696068045341e-006    0.0000000000000000e+000 ; 
    4.8400000000002369e-004    -2.8091864633207001e-006    0.0000000000000000e+000 ; 
    4.8600000000002396e-004    -2.7924206088903535e-006    0.0000000000000000e+000 ; 
    4.8800000000002422e-004    -2.7664468486664231e-006    0.0000000000000000e+000 ; 
    4.9000000000002449e-004    -2.7212302791282477e-006    0.0000000000000000e+000 ; 
    4.9200000000002475e-004    -2.6889839541824823e-006    0.0000000000000000e+000 ; 
    4.9400000000002502e-004    -2.6603626402424348e-006    0.0000000000000000e+000 ; 
    4.9600000000002528e-004    -2.6307852805410612e-006    0.0000000000000000e+000 ; 
    4.9800000000002555e-004    -2.6115147297519055e-006    0.0000000000000000e+000 ; 
    5.0000000000002581e-004    -2.5909588911325940e-006    0.0000000000000000e+000
];
