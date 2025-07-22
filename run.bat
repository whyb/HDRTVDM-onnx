REM conda create --name HDRTVDM python=3.10 -y
conda activate HDRTVDM
pip install -r requirements.txt


:: test
python method/test.py 0.jpg

python method/test.py -out_format tif *.png
python method/test.py -out_format tif *.jpg
python method/test.py -out_format tif *.jpeg
python method/test.py -out_format tif *.webp


ffmpeg -y -loop 1 -t 3 -r 1 -i 0.jpg -c:v libx265 -pix_fmt yuv420p10le  0_SDR.mp4
ffmpeg -y -loop 1 -t 3 -r 1 -i 2.jpg -c:v libx265 -pix_fmt yuv420p10le  2_SDR.mp4

ffmpeg -y -loop 1 -t 3 -r 1 -i 0_HDR.tif -c:v libx265 -pix_fmt yuv420p10le -x265-params "hdr-opt=1:colorprim=9:transfer=16:colormatrix=9:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400" 0_HDR.mp4

ffmpeg -y -loop 1 -t 3 -r 1 -i 2_HDR.tif -c:v libx265 -pix_fmt yuv420p10le -x265-params "hdr-opt=1:colorprim=9:transfer=16:colormatrix=9:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400" 2_HDR.mp4


python method/export_onnx.py --output TriSegNet.onnx --height 1080 --width 1920
python method/export_onnx.py --output TriSegNet_3DM.onnx --height 1080 --width 1920
python method/export_onnx.py --output TriSegNet_DaVinci.onnx --height 1080 --width 1920



ffmpeg -y -loop 1 -t 3 -r 1 -i 0.jpg -c:v libx265 -pix_fmt yuv420p10le -tag:v hvc1 -movflags faststart 0_SDR.mov
ffmpeg -y -loop 1 -t 3 -r 1 -i 2.jpg -c:v libx265 -pix_fmt yuv420p10le -tag:v hvc1 -movflags faststart 2_SDR.mov

ffmpeg -y -loop 1 -t 3 -r 1 -i 0_HDR.tif -c:v libx265 -pix_fmt yuv420p10le -x265-params "colorprim=9:transfer=18:colormatrix=9" -tag:v hvc1 -movflags faststart 0_HDR.mov

ffmpeg -y -loop 1 -t 3 -r 1 -i 2_HDR.tif -c:v libx265 -pix_fmt yuv420p10le -x265-params "colorprim=9:transfer=18:colormatrix=9" -tag:v hvc1 -movflags faststart 2_HDR.mov

ffmpeg -i "0_HDR.tif" -frames:v 1 -c:v libaom-av1 -pix_fmt yuv420p10le -qp 32 -vf "setparams=color_primaries=bt2020:transfer_characteristics=smpte2084:colorspace=bt2020nc,setparams=display_mastering=G(0.2650,0.6900)B(0.1500,0.0600)R(0.6800,0.3200)WP(0.3127,0.3290)L(1000.0,0.0001):cll=1000,400" 0_HDR_tmp.avif

ffmpeg -i "0_HDR.tif" -frames:v 1 -c:v libaom-av1 -pix_fmt yuv420p10le -qp 32 -vf "setparams=color_primaries=bt2020:transfer_characteristics=smpte2084:colorspace=bt2020nc" 0_HDR_tmp.avif

ffmpeg -framerate 1 -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc -color_range tv -i 0_HDR.tif -c:v libsvtav1 -pix_fmt yuv420p10le -r 1 -crf 16 -svtav1-params "enable-hdr=1:mastering-display=G(0.265,0.69)B(0.15,0.06)R(0.68,0.32)WP(0.3127,0.329)L(4000.0,0.005):content-light=1000,200:color-primaries=9:transfer-characteristics=16:matrix-coefficients=9" 0_HDR_tmp.avif