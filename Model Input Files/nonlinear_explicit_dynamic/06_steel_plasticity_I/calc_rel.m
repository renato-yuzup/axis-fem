len = 251;
v = abaqus_vy(:,2);  % Valor esperado
u = zeros(len,1);   % Valor em teste
t = abaqus_vy(:,1);
u(2:len) = plastic_test(:,7);
%u = plastic_test(:,8);
menor_valor = 5.1e-160;

% Remover valores insignificantes
diff = abs(u-v);
for i = 1 : len
    if abs(diff(i)) < menor_valor
        diff(i) = 0;
    end
end

rel = zeros(len,1);
for i = 1 : len
    %if abs(v(i)) < menor_valor
    %    rel(i) = abs(diff(i));
    %else
    if v(i) == 0 
        rel(i) = abs(diff(i));
    else
        rel(i) = abs(diff(i) / v(i));
    end
    if rel(i) >= 0.5
        rel(i) = 0;
    end
end
%rel(1:20)=0;
max(rel)
min(rel)
mean(rel)



vint = trapz(t, v);
uint = trapz(t, u);
diff = abs((uint - vint) / vint)
