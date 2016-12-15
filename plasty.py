#!/usr/bin/env python
import cv2 as cv
import click
import numpy as np


@click.command()
@click.option('--filename',
              type=click.Path(exists=True),
              required=True)
def main(filename):
    img = cv.imread(filename)
    cv.imshow('original', img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (5, 5), 2)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv.erode(img_gray, kernel, iterations=4)
    cv.imshow('eroded', eroded)
    _, filled = cv.threshold(eroded, 110, 255, cv.THRESH_BINARY)
    _, empty = cv.threshold(eroded, 50, 255, cv.THRESH_BINARY_INV)
    kernel = np.ones((7, 7), np.uint8)
    filled = cv.dilate(filled, kernel, iterations=1)
    empty = cv.dilate(empty, np.ones((3, 3), np.uint8), iterations=1)

    cv.imshow('filled mask', filled)
    cv.imshow('empty mask', empty)

    cv.imshow('gray', img_gray)

    img_filled = cv.bitwise_and(img, img, mask=filled)
    img_empty = cv.bitwise_and(img, img, mask=empty)
    cv.imshow('filled', img_filled)
    cv.imwrite('filled.png', img_filled)
    cv.imshow('empty', img_empty)
    cv.imwrite('empty.png', img_empty)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
