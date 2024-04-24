# Reliable-loc
Reliable point cloud global localization using geometric verification and pose uncertainty. The implementation is based on [PatchAugNet](https://github.com/WHU-USI3DV/PatchAugNet) and [Overlap-loc](https://github.com/PRBonn/overlap_localization)

Source code is coming soon！

# Demo (Click it to see the complete video on YouTube)
[![demo](./images/demo.gif)](https://youtu.be/-2Iiuy2haCY)

# Experimental results of Reliable-loc on 6 data
## CS college
![cs_college](./images/cs_college.gif)

## Info campus
![info_campus](./images/info_campus.gif)

## Zhongshan park
![zhongshan_park](./images/zhongshan_park.gif)

## Jiefang road
![jiefang_road](./images/jiefang_road.gif)

## Yanjiang road 1
![yanjiang_road1](./images/yanjiang_road1.gif)

## Yanjiang road 2
![yanjiang_road2](./images/yanjiang_road2.gif)

# Ablation study: the necessity of switching loc modes
## Clip 1: incomplete map coverage
<table>
  <tr>
    <td>
      <img src="./images/clip1-reg-loc.gif" alt="clip1-reg-loc" width="450">
      <strong>Reg-loc</strong>
    </td>
    <td>
      <img src="./images/clip1-reliable-loc.gif" alt="clip1-reliable-loc" width="450">
      <strong>Reliable-loc</strong>
    </td>
  </tr>
</table>

## Clip 2: feature insufficiency
<table>
  <tr>
    <td>
      <img src="./images/clip2-reg-loc.gif" alt="clip2-reg-loc" width="450">
      <strong>Reg-loc</strong>
    </td>
    <td>
      <img src="./images/clip2-reliable-loc.gif" alt="clip2-reliable-loc" width="450">
      <strong>Reliable-loc</strong>
    </td>
  </tr>
</table>

## Clip 3: feature insufficiency
<table>
  <tr>
    <td>
      <img src="./images/clip3-reg-loc.gif" alt="clip3-reg-loc" width="450">
      <strong>Reg-loc</strong>
    </td>
    <td>
      <img src="./images/clip3-reliable-loc.gif" alt="clip3-reliable-loc" width="450">
      <strong>Reliable-loc</strong>
    </td>
  </tr>
</table>

## Clip 4: feature insufficiency
<table>
  <tr>
    <td>
      <img src="./images/clip4-reg-loc.gif" alt="clip4-reg-loc" width="450">
      <strong>Reg-loc</strong>
    </td>
    <td>
      <img src="./images/clip4-reliable-loc.gif" alt="clip4-reliable-loc" width="450">
      <strong>Reliable-loc</strong>
    </td>
  </tr>
</table>

**Note**: Both λ and sigma being zero means that registration based on local features is invalid.
