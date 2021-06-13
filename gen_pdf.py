from fpdf import FPDF 

pdf = FPDF(orientation = 'P' , unit  = 'mm'  , format = 'Letter')
pdf_w = 210
psd_h = 297


pdf.add_page()
pdf.set_font('Arial' , 'B' , 16)
pdf.cell(40,10, 'Hello World!')
pdf.cell(60, 10, 'Powered by FPDF.', 0, 1, 'C')
pdf.output('outputs/test.pdf' , 'F')

