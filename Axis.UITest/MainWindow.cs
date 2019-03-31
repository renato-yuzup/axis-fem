using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

namespace Axis.UITest
{
    public partial class MainWindow : Form
    {
        private bool inputFileDirty = false;
        private bool includeFolderManualInput = false;
        private bool outputFolderManualInput = false;

        public MainWindow()
        {
            InitializeComponent();
            cboHardwareSelect.SelectedIndex = 0;
        }

        private void InputFile_TextChanged(object sender, EventArgs e)
        {
            inputFileDirty = true;
        }

        private void UpdateFolderInformation(string inputFile)
        {
            string parentDir = Path.GetDirectoryName(inputFile);
            if (!includeFolderManualInput)
            {
                txtIncludeFolder.Text = parentDir;
                includeFolderManualInput = false;
            }
            if (!outputFolderManualInput)
            {
                txtOutputFolder.Text = parentDir;
                outputFolderManualInput = false;
            }
        }

        private void InputFile_OnValidate(object sender, CancelEventArgs e)
        {
            var txt = sender as TextBox;
            bool validFile = false;
            if (txt.Text.Length == 0) return;
            {
            }
            try
            {
                validFile = File.Exists(txt.Text);
            }
            catch (System.Exception ex)
            {  
            	// ignore
            }
            if (!validFile)
            {
                return;
            }
            if (inputFileDirty)
            {
                UpdateFolderInformation(txt.Text);
            }
            inputFileDirty = false;
        }

        void AddAnalysisFlag(string flag)
        {
            flag = flag.Trim();
            if (flag.Length == 0) return;
            if (lstFlags.Items.Contains(flag)) return;
            lstFlags.Items.Add(flag);
        }

        private void txtIncludeFolder_TextChanged(object sender, EventArgs e)
        {
            var txt = sender as TextBox;
            includeFolderManualInput = (txt.Text.Trim().Length > 0);
        }

        private void txtOutputFolder_TextChanged(object sender, EventArgs e)
        {
            var txt = sender as TextBox;
            outputFolderManualInput = (txt.Text.Trim().Length > 0);
        }

        private void cmdBrowseInputFile_Click(object sender, EventArgs e)
        {
            dlgBrowseInputFile.FileName = txtInputFile.Text;
            if (dlgBrowseInputFile.ShowDialog(this) == DialogResult.OK)
            {
                txtInputFile.Text = dlgBrowseInputFile.FileName;
                UpdateFolderInformation(dlgBrowseInputFile.FileName);
            }
        }

        private void cmdRemoveFlag_Click(object sender, EventArgs e)
        {
            int selIndex = lstFlags.SelectedIndex;
            if (selIndex == -1) return;
            string removedItem = lstFlags.SelectedItem as string;
            txtFlag.Text = removedItem;
            lstFlags.Items.RemoveAt(selIndex);
            txtFlag.Focus();
        }

        private void cmdAddFlag_Click(object sender, EventArgs e)
        {
            AddAnalysisFlag(txtFlag.Text);
            txtFlag.Clear();
            txtFlag.Focus();
        }

        private void txtFlag_OnFocus(object sender, EventArgs e)
        {
            this.AcceptButton = cmdAddFlag;
        }

        private void txtFlag_OnLeave(object sender, EventArgs e)
        {
            this.AcceptButton = null;
        }

        private void cmdExit_Click(object sender, EventArgs e)
        {
            if (MessageBox.Show("Analysis has not been submitted. OK to exit?", 
                "Quit confirmation", MessageBoxButtons.YesNo, MessageBoxIcon.Question, 
                MessageBoxDefaultButton.Button2) == DialogResult.Yes)
            {
                Close();
            }
        }
    }
}
