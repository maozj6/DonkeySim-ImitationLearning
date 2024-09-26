import numpy as np
import cv2
import os
import random


def add_salt_pepper_noise(image, amount=0.05):
    noisy_image = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.5)
    salt_coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255

    num_pepper = np.ceil(amount * image.size * 0.5)
    pepper_coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    return noisy_image

def add_gaussian_noise(image, mean=0, std=25):
    noisy_image = np.copy(image)
    h, w, c = noisy_image.shape
    noise = np.random.normal(mean, std, (h, w, c))
    noisy_image = np.clip(noisy_image + noise, 0, 255)
    return noisy_image.astype(np.uint8)

def add_shot_noise(image, scale=0.1):
    noisy_image = np.copy(image)
    noise = np.random.poisson(scale, image.shape[:2])
    noisy_image = np.clip(noisy_image + noise[..., np.newaxis], 0, 255)
    return noisy_image.astype(np.uint8)

# def _add_raindrops_1_bad(image, density=0.1, length=30, thickness=1):
#     h, w, _ = image.shape
#     num_drops = int(density * h * w)
#     for _ in range(num_drops):
#         x = np.random.randint(0, w)
#         y = np.random.randint(0, h)
#         x_end = x
#         y_end = min(y + length, h - 1)
#         cv2.line(image, (x, y), (x_end, y_end), (255, 255, 255), thickness)
#     return image


# def generate_random_lines(imshape, slant, drop_length):
#     drops = []
#     for i in range(500):  ## If You want heavy rain, try increasing this
#         if slant < 0:
#             x = np.random.randint(slant, imshape[1])
#         else:
#             x = np.random.randint(0, imshape[1] - slant)
#         y = np.random.randint(0, imshape[0] - drop_length)
#         drops.append((x, y))
#     return drops
err_rain_slant="Numeric value between -20 and 20 is allowed"
err_rain_width="Width value between 1 and 5 is allowed"
err_rain_length="Length value between 0 and 100 is allowed"
def generate_random_lines(imshape,slant,drop_length,rain_type):
    drops=[]
    area=imshape[0]*imshape[1]
    no_of_drops=area//600

    if rain_type.lower()=='drizzle':
        no_of_drops=area//770
        drop_length=10
    elif rain_type.lower()=='heavy':
        drop_length=30
    elif rain_type.lower()=='torrential':
        no_of_drops=area//500
        drop_length=60

    for i in range(no_of_drops): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops,drop_length

def rain_process(image,slant,drop_length,drop_color,drop_width,rain_drops,blur_level):
    imshape = image.shape
    image_t= image.copy()
    for rain_drop in rain_drops:
        cv2.line(image_t,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    image= cv2.blur(image_t,(blur_level,blur_level)) ## rainy view are blurry
    brightness_coefficient = 0.7 ## rainy days are usually shady
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  # Conversion to HLS
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    image_RGB= cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  # Conversion to RGB
    return image_RGB

##rain_type='drizzle','heavy','torrential'
def add_rain(image,slant=-1,drop_length=20,drop_width=1,drop_color=(200,200,200),rain_type='None', blur_level=3): ## (200,200,200) a shade of gray
    slant_extreme=slant


    if isinstance(image, list):
        image_RGB=[]
        image_list=image
        imshape = image[0].shape
        if slant_extreme==-1:
            slant= np.random.randint(-10,10) ##generate random slant if no slant value is given
        rain_drops,drop_length= generate_random_lines(imshape,slant,drop_length,rain_type)
        for img in image_list:
            output= rain_process(img,slant_extreme,drop_length,drop_color,drop_width,rain_drops,blur_level)
            image_RGB.append(output)
    else:
        imshape = image.shape
        if slant_extreme==-1:
            slant= np.random.randint(-10,10) ##generate random slant if no slant value is given
        rain_drops,drop_length= generate_random_lines(imshape,slant,drop_length,rain_type)
        output= rain_process(image,slant_extreme,drop_length,drop_color,drop_width,rain_drops,blur_level)
        image_RGB=output

    return image_RGB



# def add_rain(image):
#     """
#     ref: https://www.freecodecamp.org/news/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f/
#     :param image:
#     :return:
#     """
#     imshape = image.shape
#     slant_extreme = 10
#     slant = np.random.randint(-slant_extreme, slant_extreme)
#     drop_length = 20
#     drop_width = 2
#     drop_color = (200, 200, 200)  # a shade of gray
#
#     rain_drops = generate_random_lines(imshape, slant, drop_length)
#
#     for rain_drop in rain_drops:
#         cv2.line(image, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length), drop_color,
#                  drop_width)
#
#     image = cv2.blur(image, (7, 7))  # rainy view are blurry
#
#     brightness_coefficient = 0.7  # rainy days are usually shady
#     image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  # Conversion to HLS
#     image_HLS[:, :, 1] = image_HLS[:, :, 1] * brightness_coefficient  # scale pixel values down for channel 1(Lightness)
#     image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  # Conversion to RGB
#
#     return image_RGB

# def add_snow(image, density=0.01):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     snowflakes = np.random.choice([0, 1], size=gray_image.shape, p=[1 - density, density])
#     snowflakes = (snowflakes * 255).astype(np.uint8)
#     snowflakes = cv2.bitwise_and(image, image, mask=snowflakes)
#     return cv2.add(image, cv2.cvtColor(snowflakes, cv2.COLOR_GRAY2BGR))

# err_snow_coeff="Snow coeff can only be between 0 and 1"
# def snow_process(image,snow_coeff):
#     image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
#     image_HLS = np.array(image_HLS, dtype = np.float64)
#     brightness_coefficient = 2.5
#     imshape = image.shape
#     snow_point=snow_coeff ## increase this for more snow
#     image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
#     image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
#     image_HLS = np.array(image_HLS, dtype = np.uint8)
#     image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
#     return image_RGB
#
#
# def add_snow(image, snow_coeff=-1):
#     # verify_image(image)
#     if(snow_coeff!=-1):
#         if(snow_coeff<0.0 or snow_coeff>1.0):
#             raise Exception(err_snow_coeff)
#     else:
#         snow_coeff=random.uniform(0,1)
#     snow_coeff*=255/2
#     snow_coeff+=255/3
#     if isinstance(image, list):
#         image_RGB=[]
#         image_list=image
#         for img in image_list:
#             output= snow_process(img,snow_coeff)
#             image_RGB.append(output)
#     else:
#         output= snow_process(image,snow_coeff)
#         image_RGB=output
#
#     return image_RGB


def add_snow(image, snow_point=140, brightness_coefficient=2.5):
    # 將圖像轉換為HLS色彩空間
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # 將圖像轉換為浮點數格式
    image_HLS = np.array(image_HLS, dtype=np.float64)



    # 對通道1（亮度）進行亮度調整
    image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] *= brightness_coefficient
    # 將超過255的值設置為255
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255

    # 將圖像轉換為無符號8位整數格式
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    # 將圖像轉換回RGB色彩空間
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)

    return image_RGB


def change_brightness(image, brightness_coefficient):
    """
    ref: https://www.freecodecamp.org/news/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f/
    :param image:
    :return:
    """
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64)
    # random_brightness_coefficient = np.random.uniform()+0.5 ## generates value between 0.5 and 1.5
    coefficient = brightness_coefficient
    image_HLS[:,:,1] = image_HLS[:,:,1]*coefficient ## scale pixel values up or down for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB


err_fog_coeff="Fog coeff can only be between 0 and 1"
def add_blur(image, x,y,hw,fog_coeff):
    overlay= image.copy()
    output= image.copy()
    alpha= 0.08*fog_coeff
    rad= hw//2
    point=(x+hw//2, y+hw//2)
    cv2.circle(overlay,point, int(rad), (255,255,255), -1)
    cv2.addWeighted(overlay, alpha, output, 1 -alpha ,0, output)
    return output


def generate_random_blur_coordinates(imshape,hw):
    blur_points=[]
    midx= imshape[1]//2-2*hw
    midy= imshape[0]//2-hw
    index=1
    while midx>-hw or midy>-hw:
        for i in range(hw//10*index):
            x= np.random.randint(midx,imshape[1]-midx-hw)
            y= np.random.randint(midy,imshape[0]-midy-hw)
            blur_points.append((x,y))
        midx-=3*hw*imshape[1]//sum(imshape)
        midy-=3*hw*imshape[0]//sum(imshape)
        index+=1
    return blur_points

def add_fog(image, fog_coeff=-1.0):
    if fog_coeff!=-1.0:
        if fog_coeff<0.0 or fog_coeff>1.0:
            raise Exception(err_fog_coeff)
    if isinstance(image, list):
        image_RGB=[]
        image_list=image
        imshape = image[0].shape

        for img in image_list:
            if fog_coeff==-1.0:
                fog_coeff_t=random.uniform(0.3,1)
            else:
                fog_coeff_t=fog_coeff
            hw=int(imshape[1]//3*fog_coeff_t)
            haze_list= generate_random_blur_coordinates(imshape,hw)
            for haze_points in haze_list:
                img= add_blur(img, haze_points[0],haze_points[1], hw,fog_coeff_t) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
            img = cv2.blur(img ,(hw//10,hw//10))
            image_RGB.append(img)
    else:
        imshape = image.shape
        if fog_coeff==-1:
            fog_coeff_t=random.uniform(0.3,1)
        else:
            fog_coeff_t=fog_coeff
        hw=int(imshape[1]//3*fog_coeff_t)
        haze_list= generate_random_blur_coordinates(imshape,hw)
        for haze_points in haze_list:
            image= add_blur(image, haze_points[0],haze_points[1], hw,fog_coeff_t)
        image = cv2.blur(image ,(hw//10,hw//10))
        image_RGB = image

    return image_RGB


def adjust_contrast(image, alpha=1.0, beta=0):
    """
    ref: https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
    :param beta:
    :param alpha:
    :param image:
    :return:
    """
    # Initialize values
    new_image = np.zeros(image.shape, image.dtype)
    # alpha = 1.0  # Simple contrast control
    # beta = 0  # Simple brightness control


    # print(' Basic Linear Transforms ')
    # print('-------------------------')
    # try:
    #     alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    #     beta = int(input('* Enter the beta value [0-100]: '))
    # except ValueError:
    #     print('Error, not a number')

    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    # but we wanted to show you how to access the pixels :)

    ## Method 1:
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
    ## Method 2:
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image


def add_defocus(image, kernel_size):
    # 使用高斯模糊產生虛焦效果
    blurred_image = cv2.GaussianBlur(image,
                                    (kernel_size, kernel_size),
                                     0)
    return blurred_image


def add_glass_blur(image, kernel_size):
    """
    - image (numpy.ndarray): Image loaded using cv2.
    - kernel_size (int): Size of the kernel for the glass blur effect.
    Returns: - numpy.ndarray: Image with glass blur effect.
    """
    # Apply glass blur effect
    glass_blur_img = cv2.blur(image, (kernel_size, kernel_size))
    return glass_blur_img


def add_motion_blur(image, kernel_size, style=""):
    """
    ref: https://www.geeksforgeeks.org/python-opencv-filter2d-function/
    :param image:
    :param kernel_size:
    :return:
    """
    # Specify the kernel size.
    # The greater the size, the more the motion.
    # kernel_size = 30

    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    if "vert" in style:
        # Apply the vertical kernel.
        vertical_mb = cv2.filter2D(image, -1, kernel_v)
        return vertical_mb
    elif "hori" in style:
        # Apply the horizontal kernel.
        horizontal_mb = cv2.filter2D(image, -1, kernel_h)
        return horizontal_mb
    else:
        _msg = f"Style '{style}' not found!"
        raise Exception(_msg)


def main(img_name):
    img = cv2.imread(img_name)
    noisy_img1 = add_salt_pepper_noise(img)
    noisy_img2 = add_gaussian_noise(img)
    scale = 100.0
    noisy_img3 = add_shot_noise(img, scale=scale)
    rain_type = "torrential"
    noisy_img4 = add_rain(img, rain_type=rain_type)
    # 增加亮度係數
    brightness_coefficient = 2.5
    # 雪點閾值
    snow_point = 200  # 增加此值可增加雪點數量
    noisy_img5 = add_snow(img, snow_point, brightness_coefficient)
    # between 0.5 and 1.5
    brightness_coefficient = 1.5
    noisy_img6 = change_brightness(img, brightness_coefficient)
    brightness_coefficient = 0.1
    night_img = change_brightness(img, brightness_coefficient)
    fog_coeff = 0.9
    noisy_img7 = add_fog(img, fog_coeff)
    alpha = 1.8  # Simple contrast control
    beta = 5  # Simple brightness control
    noisy_img8 = adjust_contrast(img, alpha, beta)  # 調整對比度
    kernel_size = 31  # 虛焦的核大小，越大虛焦效果越明顯
    noisy_img9 = add_defocus(img, kernel_size)  # 添加虛焦效果
    noisy_img10 = add_glass_blur(img, kernel_size=17)
    kernel_size = 20  # 運動模糊核大小，越大模糊效果越明顯
    style = "vert"
    noise_vertical = add_motion_blur(img, kernel_size, style)  # 添加運動模糊效果
    style = "hori"
    noisy_horizontal = add_motion_blur(img, kernel_size, style)  # 添加運動模糊效果

    cv2.imshow('Salt Pepper Noise', noisy_img1)
    cv2.imshow('Gaussian Noise', noisy_img2)
    cv2.imshow('Shot Noise', noisy_img3)
    cv2.imshow('Raindrops', noisy_img4)
    cv2.imshow('Snow', noisy_img5)
    cv2.imshow('Change Brightness', noisy_img6)
    cv2.imshow('Night', night_img)
    cv2.imshow('Add Fog', noisy_img7)
    cv2.imshow('Adjust Contrast', noisy_img8)
    cv2.imshow('Add Defocus', noisy_img9)
    cv2.imshow('Add Glass Blur', noisy_img10)
    cv2.imshow('Add Motion Blur Vertical', noise_vertical)
    cv2.imshow('Add Motion Blur Horizontal', noisy_horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    data_path = "resized_images"
    img_list = os.listdir(data_path)
    for image in img_list:
        main(os.path.join(data_path, image))
