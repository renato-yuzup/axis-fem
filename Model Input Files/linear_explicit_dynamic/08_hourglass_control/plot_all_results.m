% Load all of our datasets
test5_results_no_hgc;
test5_results_fb_hgc_0d01;
test5_results_fb_hgc_0d1;
test5_results_fb_hgc_0d125;
test5_results_fb_hgc_0d5;
test5_results_fb_hgc_2d0;
test5_results_fb_hgc_5d0;
test5_results_puso_hgc;

% Then, load ABAQUS results
abaqus_acceleration_tip;
abaqus_artificial_energy;
abaqus_displacement_tip;
abaqus_displacement_y_bottom_tip;
abaqus_displacement_y_top_tip;
abaqus_external_load_tip;
abaqus_stress_mid;
abaqus_velocity_tip;
abaqus_velocity_y_bottom_tip;
abaqus_velocity_y_top_tip;
abaqus_eas_acceleration_tip;
abaqus_eas_artificial_energy;
abaqus_eas_displacement_tip;
abaqus_eas_displacement_y_tip;
abaqus_eas_stress_mid;
abaqus_eas_velocity_tip;
abaqus_eas_velocity_y_tip;

% Colorize plots
color_abaqus = [106 90 205] / 255;
color_abaqus_eas = [250 128 114] / 255;
color_nohgc = [205 198 115] / 255;
color_fb0d1 = [205 79 57] / 255;
color_fb0d125 = [162 181 205] / 255;
color_fb0d01 = [64 224 208] / 255;
color_fb0d5 = [50 205 50] / 255;
color_fb2d0 = [255 215 0] / 255;
color_fb5d0 = [139 123 139] / 255;
color_puso = [218 112 214] / 255;
txtForeColor = [0 104 139] / 255;
txtBgColor = [255 250 205] / 255;

% Font parameters
legend_size = 9;

%% FIRST FIGURE: COMPARATIVE BETWEEN DIFFERENT HOURGLASS CONTROL ALGORITHMS ========================
f = figure;
set(f, 'Name', 'Test 8: Compressive pulse load Different Hourglass Control (HGC) algorithms comparison', 'Color', [1 1 1]);

% Plot applied load --------------------------------------------------------------------------------
subplot(2,3,1);
hold all;
t = title('Axial load applied at the free end nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Load (N)');

t = abaqus_load(:,1);
r = abaqus_load(:,2);
g = plot(t,r);
set(g, 'Color', color_abaqus);

t = results_no_hgc(:,1);
r = results_no_hgc(:,2);
g = plot(t,r);
set(g, 'Color', color_nohgc);

t = results_fb_hgc_0d125(:,1);
r = results_fb_hgc_0d125(:,2);
g = plot(t,r);
set(g, 'Color', color_fb0d125);

t = results_puso_hgc(:,1);
r = results_puso_hgc(:,2);
g = plot(t,r);
set(g, 'Color', color_puso);

l = legend({'ABAQUS (stiffness HGC)', 'Axis (without HGC)', 'Axis (F-B HGC, k = 0.125)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot axial displacement at the free end ----------------------------------------------------------
subplot(2,3,2);
hold all;
t = title('Axial displacement at the free end nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement (m)');

t = abaqus_u(:,1);
u = abaqus_u(:,2);
g = plot(t,u);
set(g, 'Color', color_abaqus);

t = results_no_hgc(:,1);
u = results_no_hgc(:,3);
g = plot(t,u);
set(g, 'Color', color_nohgc);

t = results_fb_hgc_0d125(:,1);
u = results_fb_hgc_0d125(:,3);
g = plot(t,u);
set(g, 'Color', color_fb0d125);

t = results_puso_hgc(:,1);
u = results_puso_hgc(:,3);
g = plot(t,u);
set(g, 'Color', color_puso);

l = legend({'ABAQUS (stiffness HGC)', 'Axis (without HGC)', 'Axis (F-B HGC, k = 0.125)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot axial stress on midpoint (element 6) --------------------------------------------------------
subplot(2,3,3);
hold all;
t = title('Axial stress on midpoint (element 6)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

t = abaqus_stress(:,1);
s = abaqus_stress(:,2);
g = plot(t,s);
set(g, 'Color', color_abaqus);

t = results_no_hgc(:,1);
s = results_no_hgc(:,6);
g = plot(t,s);
set(g, 'Color', color_nohgc);

t = results_fb_hgc_0d125(:,1);
s = results_fb_hgc_0d125(:,6);
g = plot(t,s);
set(g, 'Color', color_fb0d125);

t = results_puso_hgc(:,1);
s = results_puso_hgc(:,6);
g = plot(t,s);
set(g, 'Color', color_puso);

l = legend({'ABAQUS (stiffness HGC)', 'Axis (without HGC)', 'Axis (F-B HGC, k = 0.125)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot axial velocity ------------------------------------------------------------------------------
subplot(2,3,4);
hold all;
t = title('Axial velocity at the free end nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Velocity (m/s)');

t = abaqus_v(:,1);
v = abaqus_v(:,2);
g = plot(t,v);
set(g, 'Color', color_abaqus);

t = results_no_hgc(:,1);
v = results_no_hgc(:,4);
g = plot(t,v);
set(g, 'Color', color_nohgc);

t = results_fb_hgc_0d125(:,1);
v = results_fb_hgc_0d125(:,4);
g = plot(t,v);
set(g, 'Color', color_fb0d125);

t = results_puso_hgc(:,1);
v = results_puso_hgc(:,4);
g = plot(t,v);
set(g, 'Color', color_puso);

l = legend({'ABAQUS (stiffness HGC)', 'Axis (without HGC)', 'Axis (F-B HGC, k = 0.125)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot axial acceleration --------------------------------------------------------------------------
subplot(2,3,5);
hold all;
t = title('Axial acceleration at the free end nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');

t = abaqus_a(:,1);
a = abaqus_a(:,2);
g = plot(t,a);
set(g, 'Color', color_abaqus);

t = results_no_hgc(:,1);
a = results_no_hgc(:,5);
g = plot(t,a);
set(g, 'Color', color_nohgc);

t = results_fb_hgc_0d125(:,1);
a = results_fb_hgc_0d125(:,5);
g = plot(t,a);
set(g, 'Color', color_fb0d125);

t = results_puso_hgc(:,1);
a = results_puso_hgc(:,5);
g = plot(t,a);
set(g, 'Color', color_puso);

l = legend({'ABAQUS (stiffness HGC)', 'Axis (without HGC)', 'Axis (F-B HGC, k = 0.125)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot total artificial energy ---------------------------------------------------------------------
subplot(2,3,6);
hold all;
t = title('Total artificial (hourglass) energy');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Energy (J)');

t = abaqus_energy(:,1);
w = abaqus_energy(:,2);
g = plot(t,w);
set(g, 'Color', color_abaqus);

t = results_fb_hgc_0d125(:,1);
r = results_fb_hgc_0d125(:,16);
g = plot(t,r);
set(g, 'Color', color_fb0d125);

t = results_fb_hgc_0d5(:,1);
r = results_fb_hgc_0d5(:,16);
g = plot(t,r);
set(g, 'Color', color_fb0d5);

t = results_puso_hgc(:,1);
w = results_puso_hgc(:,16);
g = plot(t,w);
set(g, 'Color', color_puso);

legend({'ABAQUS', 'Axis (F-B HGC, k = 0.125)', 'Axis (F-B HGC, k = 0.5)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');


%% FIGURE 2: VERIFICATIONS OF TRANSVERSAL SPATIAL QUANTITIES BETWEEN DIFFERENT HGCs ================
f = figure;
set(f, 'Name', 'Test 5: Compressive pulse load -- Transversal spatial quantities', 'Color', [1 1 1]);

% Plot transversal displacement at the top ---------------------------------------------------------
subplot(2,2,1);
hold all;
t = title('Y-displacement (top probe)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement (m)');

t = abaqus_u_top_y(:,1);
u = abaqus_u_top_y(:,2);
g = plot(t,u);
set(g, 'Color', color_abaqus, 'LineWidth', 2);

t = results_no_hgc(:,1);
u = results_no_hgc(:,8);
g = plot(t,u);
set(g, 'Color', color_nohgc, 'LineWidth', 2);

t = results_fb_hgc_0d01(:,1);
u = results_fb_hgc_0d01(:,8);
g = plot(t,u);
set(g, 'Color', color_fb0d01);

t = results_fb_hgc_0d1(:,1);
u = results_fb_hgc_0d1(:,8);
g = plot(t,u);
set(g, 'Color', color_fb0d1);

t = results_fb_hgc_0d125(:,1);
u = results_fb_hgc_0d125(:,8);
g = plot(t,u);
set(g, 'Color', color_fb0d125);

t = results_fb_hgc_0d5(:,1);
u = results_fb_hgc_0d5(:,8);
g = plot(t,u);
set(g, 'Color', color_fb0d5);

t = results_puso_hgc(:,1);
u = results_puso_hgc(:,8);
g = plot(t,u);
set(g, 'Color', color_puso);

l = legend({'ABAQUS (stiffness HGC)', 'Axis (without HGC)', 'Axis (F-B HGC, k = 0.01)', 'Axis (F-B HGC, k = 0.1)', 'Axis (F-B HGC, k = 0.125)', 'Axis (F-B HGC, k = 0.5)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthWest');
set(l, 'FontSize', legend_size);


% Plot transversal displacement at the bottom ------------------------------------------------------
subplot(2,2,3);
hold all;
t = title('Y-displacement (bottom probe)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement (m)');

t = abaqus_u_bottom_y(:,1);
u = abaqus_u_bottom_y(:,2);
g = plot(t,u);
set(g, 'Color', color_abaqus, 'LineWidth', 2);

t = results_no_hgc(:,1);
u = results_no_hgc(:,9);
g = plot(t,u);
set(g, 'Color', color_nohgc, 'LineWidth', 2);

t = results_fb_hgc_0d01(:,1);
u = results_fb_hgc_0d01(:,9);
g = plot(t,u);
set(g, 'Color', color_fb0d01);

t = results_fb_hgc_0d1(:,1);
u = results_fb_hgc_0d1(:,9);
g = plot(t,u);
set(g, 'Color', color_fb0d1);

t = results_fb_hgc_0d125(:,1);
u = results_fb_hgc_0d125(:,9);
g = plot(t,u);
set(g, 'Color', color_fb0d125);

t = results_fb_hgc_0d5(:,1);
u = results_fb_hgc_0d5(:,9);
g = plot(t,u);
set(g, 'Color', color_fb0d5);

t = results_puso_hgc(:,1);
u = results_puso_hgc(:,9);
g = plot(t,u);
set(g, 'Color', color_puso);

l = legend({'ABAQUS (stiffness HGC)', 'Axis (without HGC)', 'Axis (F-B HGC, k = 0.01)', 'Axis (F-B HGC, k = 0.1)', 'Axis (F-B HGC, k = 0.125)', 'Axis (F-B HGC, k = 0.5)', 'Axis (Puso EAS HGC)'}, 'Location', 'NorthWest');
set(l, 'FontSize', legend_size);


% Plot transversal velocity at the top -------------------------------------------------------------
subplot(2,2,2);
hold all;
t = title('Y-velocity (top probe)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Velocity (m/s)');

t = abaqus_v_top_y(:,1);
u = abaqus_v_top_y(:,2);
g = plot(t,u);
set(g, 'Color', color_abaqus, 'LineWidth', 2);

t = results_no_hgc(:,1);
u = results_no_hgc(:,12);
g = plot(t,u);
set(g, 'Color', color_nohgc, 'LineWidth', 2);

t = results_fb_hgc_0d01(:,1);
u = results_fb_hgc_0d01(:,12);
g = plot(t,u);
set(g, 'Color', color_fb0d01);

t = results_fb_hgc_0d1(:,1);
u = results_fb_hgc_0d1(:,12);
g = plot(t,u);
set(g, 'Color', color_fb0d1);

t = results_fb_hgc_0d125(:,1);
u = results_fb_hgc_0d125(:,12);
g = plot(t,u);
set(g, 'Color', color_fb0d125);

t = results_fb_hgc_0d5(:,1);
u = results_fb_hgc_0d5(:,12);
g = plot(t,u);
set(g, 'Color', color_fb0d5);

t = results_puso_hgc(:,1);
u = results_puso_hgc(:,12);
g = plot(t,u);
set(g, 'Color', color_puso);

l = legend({'ABAQUS (stiffness HGC)', 'Axis (without HGC)', 'Axis (F-B HGC, k = 0.01)', 'Axis (F-B HGC, (k = 0.1)', 'Axis (F-B HGC, k = 0.125)', 'Axis (F-B HGC, k = 0.5)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthWest');
set(l, 'FontSize', legend_size);


% Plot transversal velocity at the bottom ----------------------------------------------------------
subplot(2,2,4);
hold all;
t = title('Y-velocity (bottom probe)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Velocity (m/s)');

t = abaqus_v_bottom_y(:,1);
u = abaqus_v_bottom_y(:,2);
g = plot(t,u);
set(g, 'Color', color_abaqus, 'LineWidth', 2);

t = results_no_hgc(:,1);
u = results_no_hgc(:,13);
g = plot(t,u);
set(g, 'Color', color_nohgc, 'LineWidth', 2);

t = results_fb_hgc_0d01(:,1);
u = results_fb_hgc_0d01(:,13);
g = plot(t,u);
set(g, 'Color', color_fb0d01);

t = results_fb_hgc_0d1(:,1);
u = results_fb_hgc_0d1(:,13);
g = plot(t,u);
set(g, 'Color', color_fb0d1);

t = results_fb_hgc_0d125(:,1);
u = results_fb_hgc_0d125(:,13);
g = plot(t,u);
set(g, 'Color', color_fb0d125);

t = results_fb_hgc_0d5(:,1);
u = results_fb_hgc_0d5(:,13);
g = plot(t,u);
set(g, 'Color', color_fb0d5);

t = results_puso_hgc(:,1);
u = results_puso_hgc(:,13);
g = plot(t,u);
set(g, 'Color', color_puso);

l = legend({'ABAQUS (stiffness HGC)', 'Axis (without HGC)', 'Axis (F-B HGC, k = 0.01)', 'Axis (F-B HGC, k = 0.1)', 'Axis (F-B HGC, k = 0.125)', 'Axis (F-B HGC, k = 0.5)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthWest');
set(l, 'FontSize', legend_size);


%% FIGURE 3: VALIDATION OF PUSO FORMULATION AGAINST ABAQUS ENHANCED HOURGLASS CONTROL ==============
f = figure;
set(f, 'Name', 'Test 5: Compressive pulse load -- EAS-based hourglass control', 'Color', [1 1 1]);

% Plot applied load --------------------------------------------------------------------------------
subplot(2,4,1);
hold all;
t = title('Axial load applied at the free end nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Load (N)');

t = abaqus_load(:,1);
r = abaqus_load(:,2);
g = plot(t,r);
set(g, 'Color', color_abaqus);

t = results_no_hgc(:,1);
r = results_no_hgc(:,2);
g = plot(t,r);
set(g, 'Color', color_nohgc, 'LineStyle', '--');

t = results_puso_hgc(:,1);
r = results_puso_hgc(:,2);
g = plot(t,r);
set(g, 'Color', color_puso);

l = legend({'ABAQUS', 'Axis (without HGC)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot axial displacement at the free end ----------------------------------------------------------
subplot(2,4,2);
hold all;
t = title('Axial displacement at the free end nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement (m)');

t = abaqus_u(:,1);
u = abaqus_u(:,2);
g = plot(t,u);
set(g, 'Color', color_abaqus, 'LineStyle', '--');

t = abaqus_eas_u(:,1);
u = abaqus_eas_u(:,2);
g = plot(t,u);
set(g, 'Color', color_abaqus_eas, 'LineWidth', 2);

t = results_fb_hgc_0d125(:,1);
u = results_fb_hgc_0d125(:,3);
g = plot(t,u);
set(g, 'Color', color_fb0d125, 'LineStyle', '--');

t = results_puso_hgc(:,1);
u = results_puso_hgc(:,3);
g = plot(t,u);
set(g, 'Color', color_puso, 'LineWidth', 2);

l = legend({'ABAQUS (stiffness HGC)', 'ABAQUS (enhanced HGC)', 'Axis (F-B HGC, k = 0.125)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot axial velocity ------------------------------------------------------------------------------
subplot(2,4,3);
hold all;
t = title('Axial velocity at the free end nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Velocity (m/s)');

t = abaqus_v(:,1);
v = abaqus_v(:,2);
g = plot(t,v);
set(g, 'Color', color_abaqus, 'LineStyle', '--');

t = abaqus_eas_v(:,1);
v = abaqus_eas_v(:,2);
g = plot(t,v);
set(g, 'Color', color_abaqus_eas, 'LineWidth', 2);

t = results_fb_hgc_0d125(:,1);
v = results_fb_hgc_0d125(:,4);
g = plot(t,v);
set(g, 'Color', color_fb0d125, 'LineStyle', '--');

t = results_puso_hgc(:,1);
v = results_puso_hgc(:,4);
g = plot(t,v);
set(g, 'Color', color_puso, 'LineWidth', 2);

l = legend({'ABAQUS (stiffness HGC)', 'ABAQUS (enhanced HGC)', 'Axis (F-B HGC, k = 0.125)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot axial acceleration --------------------------------------------------------------------------
subplot(2,4,4);
hold all;
t = title('Axial acceleration at the free end nodes');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');

t = abaqus_a(:,1);
a = abaqus_a(:,2);
g = plot(t,a);
set(g, 'Color', color_abaqus, 'LineStyle', '--');

t = abaqus_eas_a(:,1);
a = abaqus_eas_a(:,2);
g = plot(t,a);
set(g, 'Color', color_abaqus_eas, 'LineWidth', 2);

t = results_fb_hgc_0d125(:,1);
a = results_fb_hgc_0d125(:,5);
g = plot(t,a);
set(g, 'Color', color_fb0d125, 'LineStyle', '--');

t = results_puso_hgc(:,1);
a = results_puso_hgc(:,5);
g = plot(t,a);
set(g, 'Color', color_puso, 'LineWidth', 2);

l = legend({'ABAQUS (stiffness HGC)', 'ABAQUS (enhanced HGC)', 'Axis (F-B HGC, k = 0.125)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot axial stress on midpoint (element 6) --------------------------------------------------------
subplot(2,4,5);
hold all;
t = title('Axial stress on midpoint (element 6)');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Stress (Pa)');

t = abaqus_stress(:,1);
s = abaqus_stress(:,2);
g = plot(t,s);
set(g, 'Color', color_abaqus, 'LineStyle', '--');

t = abaqus_eas_stress(:,1);
s = abaqus_eas_stress(:,2);
g = plot(t,s);
set(g, 'Color', color_abaqus_eas, 'LineWidth', 2);

t = results_fb_hgc_0d125(:,1);
s = results_fb_hgc_0d125(:,6);
g = plot(t,s);
set(g, 'Color', color_fb0d125, 'LineStyle', '--');

t = results_puso_hgc(:,1);
s = results_puso_hgc(:,6);
g = plot(t,s);
set(g, 'Color', color_puso, 'LineWidth', 2);

l = legend({'ABAQUS (stiffness HGC)', 'ABAQUS (enhanced HGC)', 'Axis (F-B HGC, k = 0.125)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot Y-displacement at the free end --------------------------------------------------------------
subplot(2,4,6);
hold all;
t = title('Y-displacement at probe');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Displacement (m)');

t = abaqus_u_top_y(:,1);
u = abaqus_u_top_y(:,2);
g = plot(t,u);
set(g, 'Color', color_abaqus, 'LineStyle', '--');

t = abaqus_eas_u_y(:,1);
u = abaqus_eas_u_y(:,2);
g = plot(t,u);
set(g, 'Color', color_abaqus_eas, 'LineWidth', 2);

t = results_fb_hgc_0d125(:,1);
u = results_fb_hgc_0d125(:,8);
g = plot(t,u);
set(g, 'Color', color_fb0d125, 'LineStyle', '--');

t = results_puso_hgc(:,1);
u = results_puso_hgc(:,8);
g = plot(t,u);
set(g, 'Color', color_puso, 'LineWidth', 2);

l = legend({'ABAQUS (stiffness HGC)', 'ABAQUS (enhanced HGC)', 'Axis (F-B HGC, k = 0.125)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot Y-velocity ------------------------------------------------------------------------------
subplot(2,4,7);
hold all;
t = title('Y-velocity at probe');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Velocity (m/s)');

t = abaqus_v_top_y(:,1);
v = abaqus_v_top_y(:,2);
g = plot(t,v);
set(g, 'Color', color_abaqus, 'LineStyle', '--');

t = abaqus_eas_v_y(:,1);
v = abaqus_eas_v_y(:,2);
g = plot(t,v);
set(g, 'Color', color_abaqus_eas, 'LineWidth', 2);

t = results_fb_hgc_0d125(:,1);
v = results_fb_hgc_0d125(:,12);
g = plot(t,v);
set(g, 'Color', color_fb0d125, 'LineStyle', '--');

t = results_puso_hgc(:,1);
v = results_puso_hgc(:,12);
g = plot(t,v);
set(g, 'Color', color_puso, 'LineWidth', 2);

l = legend({'ABAQUS (stiffness HGC)', 'ABAQUS (enhanced HGC)', 'Axis (F-B HGC, k = 0.125)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);

% Plot total artificial energy ---------------------------------------------------------------------
subplot(2,4,8);
hold all;
t = title('Total artificial (hourglass) energy');
set(t, 'Color', txtForeColor, 'FontWeight', 'Bold');
set(t, 'BackgroundColor', txtBgColor);
xlabel('Time (s)');
ylabel('Energy (J)');

t = abaqus_energy(:,1);
w = abaqus_energy(:,2);
g = plot(t,w);
set(g, 'Color', color_abaqus);

t = abaqus_eas_energy(:,1);
w = abaqus_eas_energy(:,2);
g = plot(t,w);
set(g, 'Color', color_abaqus_eas, 'LineWidth', 2);

% t = results_fb_hgc_0d5(:,1);
% r = results_fb_hgc_0d5(:,2);
% g = plot(t,r);
% set(g, 'Color', color_fb0d5);

t = results_puso_hgc(:,1);
w = results_puso_hgc(:,16);
g = plot(t,w);
set(g, 'Color', color_puso);

l = legend({'ABAQUS (stiffness HGC)', 'ABAQUS (enhanced HGC)', 'Axis (Puso EAS HGC)'}, 'Location', 'SouthEast');
set(l, 'FontSize', legend_size);
