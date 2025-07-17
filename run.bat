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